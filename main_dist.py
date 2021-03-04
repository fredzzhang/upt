"""
Train and validate with distributed data parallel

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import argparse
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import pocket
from pocket.data import HICODet

from models import SpatioAttentiveGraph
from utils import custom_collate, CustomisedDLE, DataFactory

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    trainset = DataFactory(
        name=args.dataset, partition=args.partitions[0],
        data_root=args.data_root,
        detection_root=args.train_detection_dir,
        flip=True
    )

    valset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root,
        detection_root=args.val_detection_dir
    )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )

    val_loader = DataLoader(
        dataset=valset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            valset, 
            num_replicas=args.world_size, 
            rank=rank)
    )

    # Fix random seed for model synchronisation
    torch.manual_seed(args.random_seed)

    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        human_idx = 49
        num_classes = 117
    elif args.dataset == 'vcoco':
        object_to_target = train_loader.dataset.dataset.object_to_action
        human_idx = 1
        num_classes = 24
    net = SpatioAttentiveGraph(
        object_to_target, human_idx, num_classes=num_classes,
        num_iterations=args.num_iter, postprocess=False,
        max_human=args.max_human, max_object=args.max_object,
        box_score_thresh_pre=args.box_thresh_pre,
        box_score_thresh_post=args.box_thresh_post,
        distributed=True
    )
    # Fix backbone parameters
    for p in net.backbone.parameters():
        p.requires_grad = False

    if os.path.exists(args.checkpoint_path):
        print("=> Rank {}: continue from saved checkpoint".format(
            rank), args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optim_state_dict = checkpoint['optim_state_dict']
        sched_state_dict = checkpoint['scheduler_state_dict']
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
    else:
        print("=> Rank {}: start from a randomly initialised model".format(rank))
        optim_state_dict = None
        sched_state_dict = None
        epoch = 0; iteration = 0

    engine = CustomisedDLE(
        net,
        train_loader,
        val_loader,
        num_classes=num_classes,
        print_interval=args.print_interval,
        cache_dir=args.cache_dir
    )
    net_params = [
        p for p in engine.fetch_state_key('net').parameters()
        if p.requires_grad
    ]
    optim = torch.optim.AdamW(
        net_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=args.milestones, gamma=args.lr_decay
    )
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    engine.update_state_key(epoch=epoch, iteration=iteration)

    engine(args.num_epochs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--train-detection-dir', default='hicodet/detections/train2015', type=str)
    parser.add_argument('--val-detection-dir', default='hicodet/detections/test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--num-epochs', default=15, type=int)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=4, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--lr-decay', default=0.1, type=float,
                        help="The multiplier by which the learning rate is reduced")
    parser.add_argument('--box-thresh-pre', default=0.2, type=float)
    parser.add_argument('--box-thresh-post', default=0.2, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--milestones', nargs='+', default=[10,], type=int,
                        help="The epoch number when learning rate is reduced")
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--print-interval', default=2000, type=int)
    parser.add_argument('--checkpoint-path', default='', type=str)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    print(args)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))

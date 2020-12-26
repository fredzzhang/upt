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
from utils import custom_collate, CustomisedDataset, CustomisedDLE

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    trainset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/train2015"),
        anno_file=os.path.join(args.data_root,
            "instances_train2015.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )

    testset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/test2015"),
        anno_file=os.path.join(args.data_root,
            "instances_test2015.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )

    train_loader = DataLoader(
        dataset=CustomisedDataset(trainset,
            detection_dir=args.train_detection_dir,
            human_idx=49, 
            box_score_thresh_h=args.human_thresh,
            box_score_thresh_o=args.object_thresh,
            flip=True
        ), collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )

    val_loader = DataLoader(
        dataset=CustomisedDataset(testset,
            detection_dir=args.val_detection_dir,
            human_idx=49,
            box_score_thresh_h=args.human_thresh,
            box_score_thresh_o=args.object_thresh
        ), collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            testset, 
            num_replicas=args.world_size, 
            rank=rank)
    )

    # Fix random seed for model synchronisation
    torch.manual_seed(args.random_seed)

    net = SpatioAttentiveGraph(
        trainset.object_to_verb, 49,
        num_iterations=args.num_iter,
        postprocess=False
    )
    # Fix backbone parameters
    for p in net.backbone.parameters():
        p.requires_grad = False

    if os.path.exists(args.model_path):
        net.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        print("Model loaded from ", args.model_path)

    engine = CustomisedDLE(
        net,
        train_loader,
        val_loader,
        optim_params={
            'lr': args.learning_rate,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        },
        lr_scheduler=True,
        lr_sched_params={
            'milestones': args.milestones,
            'gamma': args.lr_decay
        },
        print_interval=args.print_interval,
        cache_dir=args.cache_dir
    )

    engine(args.num_epochs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
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
    parser.add_argument('--human-thresh', default=0.2, type=float)
    parser.add_argument('--object-thresh', default=0.2, type=float)
    parser.add_argument('--milestones', nargs='+', default=[10,],
                        help="The epoch number when learning rate is reduced")
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--print-interval', default=2000, type=int)
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    print(args)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))

"""
Train and validate with a single GPU

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import time
import torch
import argparse
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.ops.boxes import box_iou

import pocket
from pocket.data import HICODet

from models import SpatioAttentiveGraph
from utils import custom_collate, CustomisedLE, DataFactory

def main(args):

    torch.cuda.set_device(0)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.benchmark = False

    train_loader = DataLoader(
        dataset=DataFactory(
            name=args.dataset, partition=args.partitions[0],
            data_root=args.data_root,
            detection_root=args.train_detection_dir,
            box_score_thresh_h=args.human_thresh,
            box_score_thresh_o=args.object_thresh,
            flip=True
        ), collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, shuffle=True
    )

    val_loader = DataLoader(
        dataset=DataFactory(
            name=args.dataset, partition=args.partitions[1],
            data_root=args.data_root,
            detection_root=args.val_detection_dir,
            box_score_thresh_h=args.human_thresh,
            box_score_thresh_o=args.object_thresh
        ), collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )

    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        human_idx = 49
    elif args.dataset == 'vcoco':
        object_to_target = train_loader.dataset.dataset.object_to_action
        human_idx = 1
    net = SpatioAttentiveGraph(
        object_to_target, human_idx,
        num_iterations=args.num_iter,
        postprocess=False
    )
    # Fix backbone parameters
    for p in net.backbone.parameters():
        p.requires_grad = False

    if os.path.exists(args.model_path):
        net.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        print("Model loaded from ", args.model_path)

    engine = CustomisedLE(
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
    parser.add_argument('--human-thresh', default=0.2, type=float)
    parser.add_argument('--object-thresh', default=0.2, type=float)
    parser.add_argument('--batch-size', default=4, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--lr-decay', default=0.1, type=float,
                        help="The multiplier by which the learning rate is reduced")
    parser.add_argument('--milestones', nargs='+', default=[10,], type=int,
                        help="The epoch number when learning rate is reduced")
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--print-interval', default=2000, type=int)
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    print(args)

    main(args)

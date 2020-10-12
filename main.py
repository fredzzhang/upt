import os
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader

import pocket
from pocket.data import HICODet

from models import InteractGraphNet
from utils import custom_collate, CustomisedEngine, CustomisedDataset

def train(args):

    trainset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/train2015"),
        annoFile=os.path.join(args.data_root,
            "instances_train2015.json"),
        transform=torchvision.transforms.ToTensor(),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )

    testset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/test2015"),
        annoFile=os.path.join(args.data_root,
            "instances_test2015.json"),
        transform=torchvision.transforms.ToTensor(),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )

    net = InteractGraphNet(
        trainset.object_to_verb, 49,
        num_iterations=1
    )
    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        net.load_state_dict(torch.load(args.model_path)['model_state_dict'])

    engine = CustomisedEngine(
        net,
        DataLoader(CustomisedDataset(
            trainset, os.path.join(args.data_root,
                "fasterrcnn_resnet50_fpn_detections/train2015")
        ), collate_fn=custom_collate, batch_size=8,
        shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(CustomisedDataset(
            testset, os.path.join(args.data_root,
                "fasterrcnn_resnet50_fpn_detections/test2015")
        ), collate_fn=custom_collate, batch_size=8,
        num_workers=4, pin_memory=True),
        num_classes=trainset.num_action_cls,
        optim_params={
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        lr_scheduler=True,
        lr_sched_params={
            'milestones': [6, 14],
            'gamma': 0.1
        },
        print_interval=2000,
        cache_dir=args.cache_dir
    )

    engine(args.num_epochs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root',
                        required=True,
                        type=str)
    parser.add_argument('--num-epochs',
                        type=int,
                        default=25)
    parser.add_argument('--num-box-pairs-per-image',
                        type=int,
                        default=512)
    parser.add_argument('--positive-ratio',
                        type=float,
                        default=0.25)
    parser.add_argument('--box-score-thresh',
                        type=float,
                        default=0.2)
    parser.add_argument('--model-path',
                        default='',
                        type=str)
    parser.add_argument('--cache-dir',
                        type=str,
                        default='./checkpoints')
    parser.add_argument('--masked-pool',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    train(args)

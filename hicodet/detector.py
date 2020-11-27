"""
Fine-tune Faster R-CNN on HICO-DET

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import torch
import argparse
import torchvision

from torch.utils.data import Dataset, DataLoader

import pocket
from pocket.data import HICODet

class DetectorEngine(pocket.core.LearningEngine):
    def __init__(self, net, train_loader, **kwargs):
        super().__init__(net, None, train_loader, **kwargs)
    def _on_each_iteration(self):
        self._state.output = self._state.net(*self._state.inputs, targets=self._state.targets)
        self._state.loss = sum(loss for loss in self._state.output.values())
        self._state.optimizer.zero_grad()
        self._state.loss.backward()
        self._state.optimizer.step()

class HICODetObject(Dataset):
    def __init__(self, dataset, nms_thresh=0.5):
        self.dataset = dataset
        self.nms_thresh = nms_thresh
        with open('coco91tohico80.json', 'r') as f:
            corr = json.load(f)
        self.hico2coco91 = dict(zip(corr.values(), corr.keys()))
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        boxes = torch.cat([
            target['boxes_h'],
            target['boxes_o']
        ])
        labels = torch.cat([
            49 * torch.ones_like(target['object']),
            target['object']
        ])
        # Convert HICODet object (80) indices to COCO (91) indices
        converted_labels = torch.tensor([int(self.hico2coco91[i.item()]) for i in labels])
        
        return [image], [dict(boxes=boxes, labels=converted_labels)]

def collate_fn(batch):
    images = []
    targets = []
    for im, tar in batch:
        images += im
        targets += tar
    return images, targets

def main(args):

    torch.cuda.set_device(0)
    torch.manual_seed(args.random_seed)

    train_loader = DataLoader(
        dataset=HICODetObject(HICODet(
            root="hico_20160224_det/images/train2015",
            annoFile="instances_train2015.json",
            transform=torchvision.transforms.ToTensor(),
            target_transform=pocket.ops.ToTensor(input_format='dict'))),
        num_workers=4, collate_fn=collate_fn,
        shuffle=True, batch_size=args.batch_size
    )

    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    net.cuda()
    
    engine = DetectorEngine(
        net, train_loader,
        print_interval=args.print_interval,
        cache_dir=args.cache_dir,
        optim_params=dict(
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        ),
        lr_scheduler=True,
        lr_sched_params=dict(
            milestones=args.milestones,
            gamma=args.lr_decay
        )
    )

    engine(args.num_epochs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on HICO-DET")
    parser.add_argument('--num-epochs', default=20, type=int)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=0.0025, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--milestones', nargs='+', default=[10, 5])
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--print-interval', default=2000, type=int)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    print(args)

    main(args)

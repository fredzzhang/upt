"""
Fine-tune Faster R-CNN on HICO-DET

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import math
import json
import copy
import torch
import bisect
import argparse
import torchvision
import numpy as np

from PIL import Image
from itertools import repeat, chain
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler

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

"""
Batch sampler that groups images by aspect ratio
https://github.com/pytorch/vision/blob/master/references/detection/group_by_aspect_ratio.py
"""

def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)

class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size    

def compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        im = Image.open(os.path.join(
            dataset.dataset._root,
            dataset.dataset.filename(i)
        ))
        width, height = im.size
        aspect_ratios.append(float(width) / float(height))
    return aspect_ratios

def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def create_aspect_ratio_groups(dataset, k=0):
    aspect_ratios = compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print("Using {} as bins for aspect ratio quantization".format(fbins))
    print("Count of instances per bin: {}".format(counts))
    return groups

def main(args):

    torch.cuda.set_device(0)
    torch.manual_seed(args.random_seed)

    dataset = HICODetObject(HICODet(
        root="hico_20160224_det/images/train2015",
        anno_file="instances_train2015.json",
        transform=torchvision.transforms.ToTensor(),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    ))
    sampler = torch.utils.data.RandomSampler(dataset)
    group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
    batch_sampler = GroupedBatchSampler(sampler, group_ids, args.batch_size)
    train_loader = DataLoader(
        dataset=dataset, batch_sampler=batch_sampler,
        num_workers=4, collate_fn=collate_fn,
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
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--milestones', nargs='+', default=[10, 16], type=int)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--print-interval', default=2000, type=int)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    print(args)

    main(args)

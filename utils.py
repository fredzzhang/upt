"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
from pocket.ops import transforms
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist

from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, HandyTimer, BoxPairAssociation, all_gather

import sys
sys.path.append('detr')
import datasets.transforms as T

def custom_collate(batch):
    images = []
    detections = []
    targets = []
    for im, det, tar in batch:
        images.append(im)
        detections.append(det)
        targets.append(tar)
    return images, detections, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.human_idx = 49
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.human_idx = 1

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.starts_with('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
        bh = target.pop['boxes_h']; bo = target.pop['boxes_o']
        # Interlace human boxes with object boxes
        target['boxes'] = torch.cat([
            torch.stack([h, o]) for h, o in zip(bh, bo)
        ])

        image, target = self.transforms(image, target)

        return image, target

def test(net, test_loader):
    testset = test_loader.dataset.dataset
    associate = BoxPairAssociation(min_iou=0.5)

    meter = DetectionAPMeter(
        600, nproc=1,
        num_gt=testset.anno_interaction,
        algorithm='11P'
    )
    net.eval()
    for batch in tqdm(test_loader):
        inputs = pocket.ops.relocate_to_cuda(batch[:-1])
        with torch.no_grad():
            output = net(*inputs)
        if output is None or len(output) == 0:
            continue

        # Batch size is fixed as 1 for inference
        assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
        output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
        target = batch[-1][0]
        # Format detections
        boxes = output['boxes']
        boxes_h = boxes[output['boxes_h']]
        boxes_o = boxes[output['boxes_o']]
        objects = output['object']
        scores = output['scores']
        verbs = output['prediction']
        interactions = torch.tensor([
            testset.object_n_verb_to_interaction[o][v]
            for o, v in zip(objects, verbs)
        ])
        # Associate detected pairs with ground truth pairs
        labels = torch.zeros_like(scores)
        unique_hoi = interactions.unique()
        for hoi_idx in unique_hoi:
            gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
            det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
            if len(gt_idx):
                labels[det_idx] = associate(
                    (target['boxes_h'][gt_idx].view(-1, 4),
                    target['boxes_o'][gt_idx].view(-1, 4)),
                    (boxes_h[det_idx].view(-1, 4),
                    boxes_o[det_idx].view(-1, 4)),
                    scores[det_idx].view(-1)
                )

        meter.append(scores, interactions, labels)

    return meter.eval()

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, train_loader, val_loader, num_classes=117, **kwargs):
        super().__init__(net, None, train_loader, **kwargs)
        self.val_loader = val_loader
        self.num_classes = num_classes

    def _on_start(self):
        self.meter = DetectionAPMeter(self.num_classes, algorithm='11P')
        self.hoi_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)
        self.intr_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)

    def _on_each_iteration(self):
        self._state.optimizer.zero_grad()
        output = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        loss_dict = output.pop()
        if loss_dict['hoi_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.loss.backward()
        self._state.optimizer.step()

        self.hoi_loss.append(loss_dict['hoi_loss'])
        self.intr_loss.append(loss_dict['interactiveness_loss'])

        self._synchronise_and_log_results(output, self.meter)

    def _on_end_epoch(self):
        timer = HandyTimer(maxlen=2)
        # Compute training mAP
        if self._rank == 0:
            with timer:
                ap_train = self.meter.eval()
        # Run validation and compute mAP
        with timer:
            ap_val = self.validate()
        # Print performance and time
        if self._rank == 0:
            print("Epoch: {} | training mAP: {:.4f}, evaluation time: {:.2f}s |"
                "validation mAP: {:.4f}, total time: {:.2f}s\n".format(
                    self._state.epoch, ap_train.mean().item(), timer[0],
                    ap_val.mean().item(), timer[1]
            ))
            self.meter.reset()
        super()._on_end_epoch()

    def _print_statistics(self):
        super()._print_statistics()
        hoi_loss = self.hoi_loss.mean()
        intr_loss = self.intr_loss.mean()
        if self._rank == 0:
            print(f"=> HOI classification loss: {hoi_loss:.4f},",
            f"interactiveness loss: {intr_loss:.4f}")
        self.hoi_loss.reset()
        self.intr_loss.reset()

    def _synchronise_and_log_results(self, output, meter):
        scores = []; pred = []; labels = []
        # Collate results within the batch
        for result in output:
            scores.append(result['scores'].detach().cpu().numpy())
            pred.append(result['prediction'].cpu().float().numpy())
            labels.append(result["labels"].cpu().numpy())
        # Sync across subprocesses
        all_results = np.stack([
            np.concatenate(scores),
            np.concatenate(pred),
            np.concatenate(labels)
        ])
        all_results_sync = all_gather(all_results)
        # Collate and log results in master process
        if self._rank == 0:
            scores, pred, labels = torch.from_numpy(
                np.concatenate(all_results_sync, axis=1)
            ).unbind(0)
            meter.append(scores, pred, labels)

    @torch.no_grad()
    def validate(self):
        meter = DetectionAPMeter(self.num_classes, algorithm='11P')
        
        self._state.net.eval()
        for batch in self.val_loader:
            inputs = pocket.ops.relocate_to_cuda(batch)
            results = self._state.net(*inputs)

            self._synchronise_and_log_results(results, meter)

        # Evaluate mAP in master process
        if self._rank == 0:
            return meter.eval()
        else:
            return None

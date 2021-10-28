"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import torchvision.ops.boxes as box_ops

from tqdm import tqdm
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

from ops import box_cxcywh_to_xyxy

import sys
sys.path.append('detr')
import datasets.transforms as T

def custom_collate(batch):
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, training=None):
        if training is None:
            training = partition.startswith('train')
        self.training = training

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

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if self.training:
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
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
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        if self.training:
            # Generate binary labels
            labels = torch.zeros(len(target['boxes_h']), 117)
            match = torch.min(
                box_ops.box_iou(target['boxes_h'], target['boxes_h']),
                box_ops.box_iou(target['boxes_o'], target['boxes_o'])
            ) >= 0.5
            x, y = torch.nonzero(
                match * target['object'].eq(target['object'].unsqueeze(1))
            ).unbind(1)
            labels[x, target['verb'][y]] = 1
            # Remove duplicated pairs
            keep = pocket.ops.batched_pnms(
                target['boxes_h'], target['boxes_o'],
                torch.ones(target['object'].size()),
                target['object'], .5
            )
            target['boxes_h'] = target['boxes_h'][keep]
            target['boxes_o'] = target['boxes_o'][keep]
            target['object'] = target['object'][keep]
            target['labels'] = labels[keep]

        image, target = self.transforms(image, target)

        return image, target

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes

    def _on_start(self):
        self.meter = DetectionAPMeter(self.num_classes, algorithm='11P')
        # self.detection_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)
        self.interaction_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

        # self.detection_loss.append(loss_dict['detection_loss'])
        self.interaction_loss.append(loss_dict['interaction_loss'])

    def _print_statistics(self):
        super()._print_statistics()
        # detection_loss = self.detection_loss.mean()
        interaction_loss = self.interaction_loss.mean()
        if self._rank == 0:
            print(
                # f"=> Detection loss: {detection_loss:.4f},",
                f"interaction loss: {interaction_loss:.4f}"
            )
        # self.detection_loss.reset()
        self.interaction_loss.reset()

    @torch.no_grad()
    def test_hico(self, dataloader):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)

        meter = DetectionAPMeter(
            600, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)
            # Skip images without detections
            if output is None or len(output) == 0:
                continue

            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects'][output['index']]
            scores = output['scores']
            verbs = output['labels']
            interactions = torch.tensor([
                dataset.object_n_verb_to_interaction[o][v]
                for o, v in zip(objects, verbs)
            ])
            # Recover target box scale
            gt_bxh = target['boxes_h']; gt_bxh = box_cxcywh_to_xyxy(gt_bxh)
            gt_bxo = target['boxes_o']; gt_bxo = box_cxcywh_to_xyxy(gt_bxo)
            h, w = target['size']
            scale_fct = torch.stack([w, h, w, h])
            gt_bxh *= scale_fct; gt_bxo *= scale_fct
            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bxh[gt_idx].view(-1, 4),
                        gt_bxo[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                        boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)

        return meter.eval()
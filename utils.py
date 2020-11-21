"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.distributed as dist

from torch.utils.data import Dataset
from torchvision.ops.boxes import box_iou

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, HandyTimer, all_gather

def custom_collate(batch):
    images = []
    detections = []
    targets = []
    for im, det, tar in batch:
        images += im
        detections += det
        targets += tar
    return images, detections, targets

class CustomisedDataset(Dataset):
    def __init__(self, dataset, detection_dir,
            # Parameters for preprocessing
            human_idx,
            box_score_thresh_h=0.3,
            box_score_thresh_o=0.3
            ):

        self.dataset = dataset
        self.detection_dir = detection_dir

        self.human_idx = human_idx
        self.box_score_thresh_h = box_score_thresh_h
        self.box_score_thresh_o = box_score_thresh_o

    def __len__(self):
        return len(self.dataset)

    def filter_detections(self, detection):
        """Perform NMS and remove low scoring examples"""

        boxes = torch.as_tensor(detection['boxes'])
        labels = torch.as_tensor(detection['labels'])
        scores = torch.as_tensor(detection['scores'])

        # Filter out low scoring human boxes
        idx = torch.nonzero(labels == self.human_idx).squeeze(1)
        keep_idx = idx[torch.nonzero(scores[idx] >= self.box_score_thresh_h).squeeze(1)]

        # Filter out low scoring object boxes
        idx = torch.nonzero(labels != self.human_idx).squeeze(1)
        keep_idx = torch.cat([
            keep_idx,
            idx[torch.nonzero(scores[idx] >= self.box_score_thresh_o).squeeze(1)]
        ])

        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        return dict(boxes=boxes, labels=labels, scores=scores)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        target['labels'] = target['verb']

        detection_path = os.path.join(
            self.detection_dir,
            self.dataset.filename(i).replace('jpg', 'json')
        )
        with open(detection_path, 'r') as f:
            detection = pocket.ops.to_tensor(json.load(f),
                input_format='dict')

        detection = self.filter_detections(detection)

        return [image], [detection], [target]


def test(net, test_loader):
    testset = test_loader.dataset.dataset

    ap_test = DetectionAPMeter(600, num_gt=testset.anno_interaction, algorithm='11P')
    net.eval()
    for batch in tqdm(test_loader):
        inputs = pocket.ops.relocate_to_cuda(batch[:-1])
        with torch.no_grad():
            output = net(*inputs)
        if output is None:
            continue

        for result, target in zip(output, batch[-1]):
            result = pocket.ops.relocate_to_cpu(result)

            box_idx = result['index']
            _, num = torch.unique(box_idx, return_counts=True)
            num = num.tolist()
            # Reformat the predicted classes and scores
            # TARGET_CLASS: [SCORE, BINARY_LABELS]
            predictions = [{
                testset.object_n_verb_to_interaction[o][v]:[s.item(), 0]
                for v, s in zip(verbs, scores)
                } for verbs, scores, o in zip(
                    result['prediction'].split(num),
                    result['scores'].split(num),
                    result['object']
            )]
            # Compute minimum IoU
            match = torch.min(
                box_iou(target['boxes_h'], result['boxes_h']),
                box_iou(target['boxes_o'], result['boxes_o'])
            ) >= 0.5

            # Associate detected box pairs with ground truth
            for i, m in enumerate(match):
                target_class = target["hoi"][i].item()
                # For each ground truth box pair, find the
                # detections with sufficient IoU
                det_idx = m.nonzero().squeeze(1)
                if len(det_idx) == 0:
                    continue
                # Retrieve the scores of matched detections
                # When target class was not predicted, fill the score as -1
                match_scores = torch.as_tensor([p[target_class][0]
                    if target_class in p else -1 for p in predictions])
                match_idx = match_scores.argmax()
                # None of matched detections have predicted target class
                if match_scores[match_idx] == -1:
                    continue
                predictions[match_idx][target_class][1] = 1

            pred = torch.cat([
                torch.Tensor(list(p.keys())) for p in predictions
            ])
            scores, labels = torch.cat([
                torch.Tensor(list(p.values())) for p in predictions
            ]).unbind(1)

            ap_test.append(scores, pred, labels)

    return ap_test.eval()

class CustomisedEngine(DistributedLearningEngine):
    def __init__(self, net, train_loader, val_loader, num_classes=117, **kwargs):
        super().__init__(net, None, train_loader, **kwargs)
        self.val_loader = val_loader
        self.num_classes = num_classes

    def _on_each_iteration(self):
        self._state.optimizer.zero_grad()
        self._state.output = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        self._state.loss = self._state.output.pop()
        self._state.loss.backward()
        self._state.optimizer.step()

        self._synchronise_and_log_results(self._state.output, self.ap_train)

    def _on_start_epoch(self):
        super()._on_start_epoch()
        if self._rank == 0:
            self.ap_train = DetectionAPMeter(self.num_classes, algorithm='11P')

    def _on_end_epoch(self):
        super()._on_end_epoch()
        timer = HandyTimer(maxlen=2)
        # Compute training mAP
        if self._rank == 0:
            with timer:
                ap_train = self.ap_train.eval()
        # Run validation and compute mAP
        with timer:
            ap_val = self.validate()
        # Print time
        if self._rank == 0:
            print("Epoch: {} | training mAP: {:.4f}, evaluation time: {:.2f}s |"
                "validation mAP: {:.4f}, total time: {:.2f}s".format(
                    self._state.epoch, ap_train.mean().item(), timer[0],
                    ap_val.mean().item(), timer[1]
            ))

    def _synchronise_and_log_results(self, output, meter):
        scores = []; pred = []; labels = []
        # Collate results within the batch
        for result in output:
            scores.append(torch.cat(result['scores']).detach().cpu().numpy())
            pred.append(torch.cat(result['predictions']).cpu().float().numpy())
            labels.append(torch.cat(result["labels"]).cpu().numpy())
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
        """Test the network and return classification mAP"""
        # Instantiate the AP meter in the master process
        if self._rank == 0:
            ap_val = DetectionAPMeter(self.num_classes, algorithm='11P')
        
        self._state.net.eval()
        for batch in self.val_loader:
            inputs = pocket.ops.relocate_to_cuda(batch)
            results = self._state.net(*inputs)
            if results is None:
                continue

            self._synchronise_and_log_results(results, ap_val)

        # Evaluate mAP in master process
        if self._rank == 0:
            return ap_val.eval()
        else:
            return None

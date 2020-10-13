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
import torch.distributed as dist

from torch.utils.data import Dataset
from torchvision.ops.boxes import box_iou

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, all_gather

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
    def __init__(self, dataset, detection_dir):
        self.dataset = dataset
        self.detection_dir = detection_dir

    def __len__(self):
        return len(self.dataset)

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

        return [image], [detection], [target]

class CustomisedEngine(DistributedLearningEngine):
    def __init__(self, net, train_loader, test_loader, num_classes, **kwargs):
        super().__init__(net, None, train_loader, **kwargs)
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.ap = dict()

    def _on_start_epoch(self):
        super()._on_start_epoch()
        # Instantiate the AP meter in the master process
        if self._rank == 0:
            self.ap_train = DetectionAPMeter(self.num_classes, algorithm="INT")

    def _on_each_iteration(self):
        self._state.optimizer.zero_grad()
        self._state.output = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        self._state.loss = self._state.output.pop()
        self._state.loss.backward()
        self._state.optimizer.step()

        # Add results to AP meter
        for result_one in self._state.output:
            result_tensor = torch.stack([
                torch.cat(result_one["scores"]),
                torch.cat(result_one["labels"]),
                torch.cat(result_one["gt_labels"])
            ])
            # Collect results across subprocesses
            scores, pred, labels = torch.cat(all_gather(result_tensor), dim=1).unbind()
            # Log results in master process
            if self._rank == 0:
                self.ap_train.append(scores, pred, labels)

    def _on_end_epoch(self):
        super()._on_end_epoch()
        
        ap_test = self.test()
        # Evaluate mAP, print message and save cache in the master process
        if self._rank == 0:
            ap_train = self.ap_train.eval()
            print("\n=>Evaluation (+{:.2f}s)\n"
                "Epoch: {} | mAP (train): {:.4f} | mAP (test): {:.4f}".format(
                    time.time() - self._dawn, self._state.epoch,
                    ap_train.mean().item(), ap_test.mean().item()
                ))
            self.ap[self._state.epoch] = dict(
                Train=ap_train.tolist(),
                Test=ap_test.tolist()
            )

            with open(os.path.join(self._cache_dir, 'ap.json'), 'w') as f:
                json.dump(self.ap, f)

    def test(self, min_iou=0.5):
        """Test the network and return classification mAP"""
        # Instantiate the AP meter in the master process
        if self._rank == 0:
            ap_test = DetectionAPMeter(self.num_classes, algorithm="INT")
        
        self._state.net.eval()
        for batch in self.test_loader:
            inputs = pocket.ops.relocate_to_device(
                batch[:-1], self._device)
            with torch.no_grad():
                results = self._state.net(*inputs)
            if results is None:
                continue

            for result, target in zip(results, batch[-1]):
                result = pocket.ops.relocate_to_cpu(result)

                # Reformat the predicted classes and scores
                # CLASS_IDX: [SCORE, BINARY_LABELS]
                predictions = [{
                    int(k.item()):[v.item(), 0]
                    for k, v in zip(l, s)
                } for l, s in zip(result["labels"], result["scores"])]
                # Compute minimum IoU and apply threshold
                match = torch.min(
                    box_iou(target['boxes_h'], result['boxes_h']),
                    box_iou(target['boxes_o'], result['boxes_o'])
                ) > min_iou

                # Associate detected box pairs with ground truth
                for i, m in enumerate(match):
                    target_class = target["labels"][i].item()
                    # For each ground truth box pair, find the 
                    # detections with sufficient IoU
                    det_idx = m.nonzero().squeeze(1)
                    if len(det_idx) == 0:
                        continue
                    # Assign classification labels
                    for d_idx in det_idx.tolist():
                        try:
                            predictions[d_idx][target_class][1] = 1
                        # Target class was not predicted in this detection
                        except KeyError:
                            continue

                pred = torch.cat([
                    torch.tensor(list(p.keys())) for p in predictions
                ])
                scores_n_labels = torch.cat([
                    torch.tensor(list(p.values())) for p in predictions
                ])
                # Collect results across subprocesses
                pred = torch.cat(all_gather(pred))
                scores, labels = torch.cat(all_gather(scores_n_labels), dim=0).unbind(1)

                # Log results in master process
                if self._rank == 0:
                    ap_test.append(scores, pred, labels)

        # Evaluate mAP in master process
        if self._rank == 0:
            return ap_test.eval()
        else:
            return None

    def eval_on_hicodet_protocol(self):
        """Evaluate the model based on HICODet protocol"""
        # NOTE Add conversion from action classes to interaction classes
        return 0

        # # Detection mAP
        # if self._rank == 0:
        #     ap_test = DetectionAPMeter(
        #         self.num_classes,
        #         num_gt=self.test_loader.dataset.anno_action,
        #         algorithm='11P'
        #     )
        # self._state.net.eval()
        # for batch in self.test_loader:
        #     inputs = pocket.ops.relocate_to_device(
        #         batch[:-1], self._device)
        #     with torch.no_grad():
        #         results = self._state.net(*inputs)
        #     if results is None:
        #         continue

        #     for result, target in zip(results, batch[-1]):
        #         result = pocket.ops.relocate_to_cpu(result)

        #         # Reformat the predicted classes and scores
        #         # CLASS_IDX: [SCORE, BINARY_LABELS]
        #         predictions = [{
        #             int(k.item()):[v.item(), 0]
        #             for k, v in zip(l, s)
        #         } for l, s in zip(result["labels"], result["scores"])]
        #         # Compute minimum IoU
        #         iou = torch.min(
        #             box_iou(target['boxes_h'], result['boxes_h']),
        #             box_iou(target['boxes_o'], result['boxes_o'])
        #         )
        #         # Assign each detection to GT with the highest IoU
        #         max_iou, max_idx = iou.max(0)
        #         match = -1 * torch.ones_like(iou)
        #         match[max_idx, torch.arange(iou.shape[1], device=iou.device)] = max_iou

        #         # Associate detected box pairs with ground truth
        #         for i, m in enumerate(match):
        #             target_class = target["labels"][i].item()
        #             # For each ground truth box pair, find the 
        #             # detections with sufficient IoU
        #             det_idx = m.nonzero().squeeze(1)
        #             if len(det_idx) == 0:
        #                 continue
        #             # Retrieve the scores of matched detections
        #             # When target class was not predicted, fill the score as -1
        #             match_scores = torch.as_tensor([p[target_class][0]
        #                 if target_class in p else -1 for p in predictions])
        #             match_idx = match_scores.argmax()
        #             # None of matched detections have predicted target class
        #             if match_scores[match_idx] == -1:
        #                 continue
        #             predictions[match_idx][target_class][2] = 1
        
        #         pred = torch.cat([
        #             torch.tensor(list(p.keys())) for p in predictions
        #         ])
        #         scores_n_labels = torch.cat([
        #             torch.tensor(list(p.values())) for p in predictions
        #         ])
        #         # Collect results across subprocesses
        #         pred = torch.cat(all_gather(pred))
        #         scores, labels = torch.cat(all_gather(scores_n_labels), dim=0).unbind(1)

        #         # Log results in master process
        #         if self._rank == 0:
        #             ap_test.append(scores, pred, labels)

        # # Evaluate mAP in master process
        # if self._rank == 0:
        #     return ap_test.eval()
        # else:
        #     return None
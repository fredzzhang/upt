"""
Evaluate generated object detections on HICODET

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.ops.boxes import batched_nms, box_iou

import pocket
from pocket.data import HICODet
from pocket.ops import to_tensor

def analyse_recall(
        dataset, detection_dir,
        h_thresh, o_thresh, nms_thresh,
        max_human, max_object,
        training, human_idx=49, min_iou=0.5
    ):
    num_pairs_object = np.zeros(80)
    meter = pocket.utils.DetectionAPMeter(
        80, num_gt=dataset.anno_object,
        algorithm='INT', nproc=10
    )
    # Skip images without valid human-object pairs
    valid_idx = dataset._idx

    num_gt_retrieved = torch.zeros(600)
    num_pair = []
    for i in tqdm(valid_idx):
        # Load annotation
        target = to_tensor(dataset._anno[i], input_format='dict')
        # Load detection
        detection_path = os.path.join(
            detection_dir,
            dataset._filenames[i].replace('jpg', 'json')
        )
        with open(detection_path, 'r') as f:
            detection = to_tensor(json.load(f), input_format='dict')

        boxes = detection['boxes']
        labels = detection['labels']
        scores = detection['scores']        
        
        # Filter out low scoring human boxes
        idx = torch.nonzero(labels == human_idx).squeeze(1)
        keep_idx = idx[torch.nonzero(scores[idx] >= h_thresh).squeeze(1)]

        # Filter out low scoring object boxes
        idx = torch.nonzero(labels != human_idx).squeeze(1)
        keep_idx = torch.cat([
            keep_idx,
            idx[torch.nonzero(scores[idx] >= o_thresh).squeeze(1)]
        ])

        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        # Class-wise non-maximum suppression
        keep_idx = batched_nms(
            boxes, scores, labels, nms_thresh
        )
        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        scores = scores[sorted_idx]
        labels = labels[sorted_idx]

        h_idx = torch.nonzero(labels == human_idx).squeeze(1)
        o_idx = torch.nonzero(labels != human_idx).squeeze(1)
        if len(h_idx) > max_human:
            h_idx = h_idx[:max_human]
        if len(o_idx) > max_object:
            o_idx = o_idx[:max_object]
        keep_idx = torch.cat([h_idx, o_idx])

        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        gt_boxes = torch.cat([
            target['boxes_h'],
            target['boxes_o']
        ])
        gt_classes = torch.cat([
            human_idx * torch.ones_like(target['object']),
            target['object']
        ])        
        # Do NMS on ground truth boxes
        keep_gt_idx = batched_nms(
            gt_boxes, torch.ones_like(gt_classes).float(), gt_classes, nms_thresh
        )
        gt_boxes = gt_boxes[keep_gt_idx]
        gt_classes = gt_classes[keep_gt_idx]
        # Update number of ground truth annotations
        for c in gt_classes:
            num_pairs_object[c] += 1

        det_labels = torch.zeros(len(keep_idx))
        match = box_iou(gt_boxes, boxes) >= min_iou
        for j, m in enumerate(match):
            match_idx = torch.nonzero(m).squeeze(1)
            if len(match_idx):
                matched_scores = -torch.ones_like(scores)
                matched_scores[match_idx] = scores[match_idx]
                for m_idx in match_idx:
                    if labels[m_idx] != gt_classes[j]:
                        matched_scores[m_idx] = -1
                intra_idx = torch.argmax(matched_scores)
                if matched_scores[intra_idx] == -1:
                    continue
                det_labels[intra_idx] = 1

        meter.append(scores, labels, det_labels)

    meter.num_gt = num_pairs_object.tolist()
    map_ = meter.eval()
    print(map_.mean(), meter.max_rec.mean())

def main(args):
    
    testset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/test2015"),
        annoFile=os.path.join(args.data_root,
            "instances_test2015.json")
    )

    h_score_thresh = args.human_thresh
    o_score_thresh = args.object_thresh
    nms_thresh = args.nms_thresh
    max_human = args.max_human
    max_object = args.max_object

    analyse_recall(testset,
        os.path.join(args.data_root,
            "fasterrcnn_resnet50_fpn_detections/test2015_finetuned"),
        h_score_thresh, o_score_thresh, nms_thresh,
        max_human, max_object, False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset size analysis")
    parser.add_argument('--data-root', required=True, type=str)
    parser.add_argument('--human-thresh', default=0.5, type=float,
                        help="Threshold used to filter low scoring human detections")
    parser.add_argument('--max-human', default=10, type=int,
                        help="Maximum number of human instances to keep in an image")
    parser.add_argument('--object-thresh', default=0.5, type=float,
                        help="Threshold used to filter low scoring object detections")
    parser.add_argument('--max-object', default=10, type=int,
                        help="Maximum number of (pure) object instances to keep in an image")
    parser.add_argument('--nms-thresh', default=0.5, type=float,
                        help="Threshold for non-maximum suppression")
    args = parser.parse_args()

    print(args)
    main(args)

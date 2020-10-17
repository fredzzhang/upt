"""
Compute the number of box pairs with respect selected thresholds
for NMS and filtering

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
    # Skip images without valid human-object pairs
    valid_idx = dataset._idx

    num_gt_retrieved = torch.zeros(600)
    num_pair = []
    for idx in tqdm(valid_idx):
        # Load annotation
        target = to_tensor(dataset._anno[idx], input_format='dict')
        # Load detection
        detection_path = os.path.join(
            detection_dir,
            dataset._filenames[idx].replace('jpg', 'json')
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

        # Append ground truth during training
        if training:
            n = target["boxes_h"].shape[0]
            boxes = torch.cat([target["boxes_h"], target["boxes_o"], boxes])
            scores = torch.cat([torch.ones(2 * n), scores])
            labels = torch.cat([
                human_idx * torch.ones(n).long(),
                target["object"],
                labels
            ])

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

        # Record box pair number
        num_pair.append(len(h_idx) * len(keep_idx))
        # Record retrieved ground truth instances
        h_retrieved = torch.sum(
            box_iou(target["boxes_h"], boxes[h_idx]) > min_iou,
            dim=1)
        o_retrieved = torch.sum(
            box_iou(target["boxes_o"], boxes[keep_idx]) > min_iou,
            dim=1)
        retrieved = (h_retrieved > 0) * (o_retrieved > 0)
        for cls_idx, is_retrieved in zip(target["hoi"], retrieved):
            num_gt_retrieved[cls_idx] += is_retrieved

    num_pair = torch.as_tensor(num_pair)
    recall = num_gt_retrieved / torch.as_tensor(dataset.anno_interaction, dtype=torch.float32)
    header = "Training set:" if training else "Test set:"
    print(header)
    print("There are {} box pairs in total, {:.2f} per image on average\n"
        "Mean maximum recall: {:.4f}".format(
            num_pair.sum(), num_pair.float().mean(), recall.mean())
    )

    string = "train" if training else "test"
    torch.save(num_pair, "{}_data.pt".format(string))

def main(args):
    trainset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/train2015"),
        annoFile=os.path.join(args.data_root,
            "instances_train2015.json")
    )
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

    analyse_recall(trainset, 
        os.path.join(args.data_root,
            "fasterrcnn_resnet50_fpn_detections/train2015"),
        h_score_thresh, o_score_thresh, nms_thresh,
        max_human, max_object, True)

    analyse_recall(testset,
        os.path.join(args.data_root,
            "fasterrcnn_resnet50_fpn_detections/test2015"),
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

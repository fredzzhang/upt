"""
Run Faster R-CNN with ResNet50-FPN on HICO-DET
Fred Zhang <frederic.zhang@anu.edu.au>
The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import argparse
import torch
import torchvision

from pocket.ops import relocate_to_cpu
from pocket.data import HICODet
from pocket.models import fasterrcnn_resnet_fpn

OBJECT_CORR = {
        1: 49, 2: 9, 3: 18, 4: 44, 5: 0, 6: 16, 7: 73, 8: 74, 9: 11,
        10: 72, 11: 31, 13: 63, 14: 48, 15: 8, 16: 10, 17: 20, 18: 28,
        19: 37, 20: 56, 21: 25, 22: 30, 23: 6, 24: 79, 25: 34, 27: 2,
        28: 76, 31: 36, 32: 68, 33: 64, 34: 33, 35: 59, 36: 60, 37: 62,
        38: 40, 39: 4, 40: 5, 41: 58, 42: 65, 43: 67, 44: 13, 46: 78,
        47: 26, 48: 32, 49: 41, 50: 61, 51: 14, 52: 3, 53: 1, 54: 54,
        55: 46, 56: 15, 57: 19, 58: 38, 59: 50, 60: 29, 61: 17, 62: 22,
        63: 24, 64: 51, 65: 7, 67: 27, 70: 70, 72: 75, 73: 42, 74: 45,
        75: 53, 76: 39, 77: 21, 78: 43, 79: 47, 80: 69, 81: 57, 82: 52,
        84: 12, 85: 23, 86: 77, 87: 55, 88: 66, 89: 35, 90: 71
}

def main(partition):
    cache_dir = './fasterrcnn_resnet50_fpn_detections/' + partition
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    dataset = HICODet(
            root='./hico_20160224_det/images/' + partition,
            annoFile='./instances_{}.json'.format(partition)
            )

    t = torchvision.transforms.ToTensor()

    detector = fasterrcnn_resnet_fpn('resnet50',
            pretrained=True, box_score_thresh=0.0, 
            box_nms_thresh=0.9,
            box_detections_per_img=1000
            ).eval().cuda()

    with torch.no_grad():
        for idx, (image, _) in enumerate(dataset):

            image = t(image).cuda()
            detections = detector([image])[0]

            detections['boxes'] = detections['boxes'].tolist()
            detections['scores'] = detections['scores'].tolist()
            labels = detections['labels'].tolist()

            remove_idx = []
            for j, obj in enumerate(labels):
                if obj in OBJECT_CORR:
                    labels[j] = OBJECT_CORR[obj]
                else:
                    remove_idx.append(j)
            detections['labels'] = labels
            # Remove detections of deprecated object classes
            remove_idx.sort(reverse=True)
            for j in remove_idx:
                detections['boxes'].pop(j)
                detections['scores'].pop(j)
                detections['labels'].pop(j)

            with open(os.path.join(cache_dir,
                    dataset.filename(idx).replace('jpg', 'json')), 'w') as f:
                json.dump(detections, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--partition',
            type=str, default='train2015')
    args = parser.parse_args()

    main(args.partition)
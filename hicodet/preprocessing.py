"""
Run Faster R-CNN with ResNet50-FPN on HICO-DET

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import torch
import argparse
import torchvision
from tqdm import tqdm

from pocket.ops import relocate_to_cpu
from pocket.data import HICODet
from pocket.models import fasterrcnn_resnet_fpn

with open('coco91tohico80.json', 'r') as f:
    OBJECT_CORR = json.load(f)

def main(args):
    cache_dir = os.path.join(args.cache_dir, args.partition)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    dataset = HICODet(
            root='./hico_20160224_det/images/' + args.partition,
            anno_file='./instances_{}.json'.format(args.partition)
            )

    t = torchvision.transforms.ToTensor()

    detector = fasterrcnn_resnet_fpn('resnet50',
            pretrained=True,
            box_score_thresh=args.score_thresh, 
            box_nms_thresh=args.nms_thresh,
            box_detections_per_img=args.num_detections_per_image
            )
    if os.path.exists(args.ckpt_path):
        detector.load_state_dict(torch.load(args.ckpt_path)['model_state_dict'])
    detector.eval()
    detector.cuda()

    with torch.no_grad():
        for idx, (image, _) in enumerate(tqdm(dataset)):

            image = t(image).cuda()
            detections = detector([image])[0]

            detections['boxes'] = detections['boxes'].tolist()
            detections['scores'] = detections['scores'].tolist()
            labels = detections['labels'].tolist()

            remove_idx = []
            for j, obj in enumerate(labels):
                if str(obj) in OBJECT_CORR:
                    labels[j] = OBJECT_CORR[str(obj)]
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
    parser.add_argument('--cache-dir',
            type=str, default='fasterrcnn_resnet50_fpn_detections')
    parser.add_argument('--nms-thresh',
            type=float, default=0.5)
    parser.add_argument('--score-thresh',
            type=float, default=0.5)
    parser.add_argument('--num-detections-per-image',
            type=int, default=100)
    parser.add_argument('--ckpt-path',
            type=str, default='',
            help="Path to a checkpoint that contains the weights for a model")
    args = parser.parse_args()

    main(args)

"""
Debug script

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import pocket
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import custom_collate, DataFactory
from models import SpatioAttentiveGraph


# @torch.no_grad()
def main(args):

    dataset = DataFactory(
        name='hicodet', partition=args.partition,
        data_root=args.data_root,
        detection_root=args.detection_dir,
        flip=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=custom_collate,
        batch_size=4, shuffle=False
    )

    net = SpatioAttentiveGraph(
        dataset.dataset.object_to_verb, 49,
        num_iterations=args.num_iter
    )
    # net.eval()

    if os.path.exists(args.model_path):
        print("\nLoading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
            "Proceed to use a randomly initialised model.\n")
    else:
        print("\nProceed with a randomly initialised model\n")

    # iterator = iter(dataloader)
    # image, detection, target = next(iterator)
    
    image, detection, target = dataset[34985]
    image = [image]; detection = [detection]; target = [target]

    output = net(image, detection, target)
    torch.save(output, 'data.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--detection-dir', default='hicodet/detections/test2015',
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--human-thresh', default=0.2, type=float)
    parser.add_argument('--object-thresh', default=0.2, type=float)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='', type=str)
    
    args = parser.parse_args()
    print(args)

    main(args)

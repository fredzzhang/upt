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

from pocket.utils import draw_boxes

def visualise_entire_image(dataset, output):
    """Visualise bounding box pairs in the whole image by classes"""
    bh=output['boxes_h']
    bo=output['boxes_o']
    no = len(bo)

    bbox, inverse = torch.unique(torch.cat([bo, bh]), dim=0, return_inverse=True)
    idxh = inverse[no:]
    idxo = inverse[:no]

    im = dataset.dataset.load_image(
        os.path.join(
            dataset.dataset._root,
            dataset.dataset.filename(args.index)
        )
    )

    # Draw the bounding boxes
    fig = plt.figure()
    plt.imshow(im)
    ax = plt.gca()
    draw_boxes(ax, bbox)
    plt.show()

    # Print predicted classes and scores
    scores = output['scores']
    prior = output['prior']
    index = output['index']
    pred = output['prediction']
    labels = output['labels']

    unique_hoi = torch.unique(pred)
    for hoi in unique_hoi:
        print(f"\n=> Interaction: {dataset.interacitons[hoi]}")
        sample_idx = torch.nonzero(pred == hoi).squeeze(1)
        for idx in sample_idx:
            b_idx = index[idx]
            print(
                f"({idxh[b_idx], idxo[b_idx]}),",
                f"score: {scores[idx]}, prior: {prior[idx]}",
                f"label: {bool(labels[idx])}"
            )

@torch.no_grad()
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
    net.eval()

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
    visualise_entire_image(dataset, output[0])
    # torch.save(output, 'data.pt')

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

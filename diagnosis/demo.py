"""
Visualise box pairs in an image

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import torch
import pocket
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff

sys.path.append('/'.join(os.path.abspath(sys.argv[0]).split('/')[:-2]))

from utils import custom_collate, DataFactory
from models import SpatiallyConditionedGraph as SCG

def colour_pool(n):
    pool = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#17becf', '#e377c2'
    ]
    nc = len(pool)

    repeat = n // nc
    big_pool = []
    for _ in range(repeat):
        big_pool += pool
    return big_pool + pool[:n%nc]


def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()

def visualise_entire_image(dataset, output):
    """Visualise bounding box pairs in the whole image by classes"""
    boxes = output['boxes']
    bh = output['boxes_h']
    bo = output['boxes_o']

    attn_maps = output['attn_maps']

    _, axe = plt.subplots(2, 4)
    axe = np.concatenate(axe)
    for ax, attn in zip(axe, attn_maps[0]):
        ax.imshow(attn)

    im = dataset.dataset.load_image(
        os.path.join(
            dataset.dataset._root,
            dataset.dataset.filename(args.index)
        )
    )

    # Print predicted classes and scores
    scores = output['scores']
    prior = output['prior']
    pred = output['prediction']
    labels = output['labels']

    unique_actions = torch.unique(pred)
    for verb in unique_actions:
        print(f"\n=> Action: {dataset.dataset.verbs[verb]}")
        sample_idx = torch.nonzero(pred == verb).squeeze(1)
        for idx in sample_idx:
            idxh = bh[idx]; idxo = bo[idx]
            print(
                f"({idxh.item():<2}, {idxo.item():<2}),",
                f"score: {scores[idx]:.4f}, prior: {prior[0, idx]:.2f}, {prior[1, idx]:.2f}",
                f"label: {bool(labels[idx])}"
            )

    # Draw the bounding boxes
    plt.figure()
    plt.imshow(im)
    ax = plt.gca()
    draw_boxes(ax, boxes)
    plt.show()

@torch.no_grad()
def main(args):

    dataset = DataFactory(
        name='hicodet', partition=args.partition,
        data_root=args.data_root,
        detection_root=args.detection_dir,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=custom_collate,
        batch_size=4, shuffle=False
    )

    net = SCG(
        dataset.dataset.object_to_verb, 49,
        num_iterations=args.num_iter,
        box_score_thresh=args.box_score_thresh
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
    
    image, detection, target = dataset[args.index]
    image = [image]; detection = [detection]; target = [target]

    output = net(image, detection, target)
    visualise_entire_image(dataset, output[0])
    # torch.save(output, 'data.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', default='../hicodet', type=str)
    parser.add_argument('--detection-dir', default='../hicodet/detections/test2015',
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--index', default=0, type=int)
    
    args = parser.parse_args()
    print(args)

    main(args)

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

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import DataFactory
from detector import build_detector

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
    # Rescale the boxes to original image size
    ow, oh = output['original_size']
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct
    nh = output['objects'].cumsum(0).eq(0).sum() + 1; no = len(output['objects']) / nh + 1

    attn_1 = output['attn_maps'][0]
    attn_2 = output['attn_maps'][1]

    # Visualise unary attention
    fig, axe = plt.subplots(2, 4)
    axe = np.concatenate(axe)
    for ax, attn in zip(axe, attn_1):
        im = ax.imshow(attn.squeeze().T, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

    x, y = torch.meshgrid(torch.arange(nh), torch.arange(no.long()))
    x, y = torch.nonzero(x != y).unbind(1)
    pairs = [str((i.item(), j.item())) for i, j in zip(x, y)]
    # Visualise pairwise attention
    fig, axe = plt.subplots(2, 4)
    axe = np.concatenate(axe)
    ticks = list(range(len(pairs)))
    for ax, attn in zip(axe, attn_2):
        im = ax.imshow(attn, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(pairs, rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels(pairs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

    im = dataset.dataset.load_image(
        os.path.join(
            dataset.dataset._root,
            dataset.dataset.filename(args.index)
        )
    )

    # Print predicted classes and scores
    scores = output['scores']
    pred = output['labels']
    pairing = output['pairing']

    unique_actions = torch.unique(pred)
    for verb in unique_actions:
        print(f"\n=> Action: {dataset.dataset.verbs[verb]}")
        sample_idx = torch.nonzero(pred == verb).squeeze(1)
        for idx in sample_idx:
            idxh, idxo = pairing[:, idx]
            print(
                f"({idxh.item():<2}, {idxo.item():<2}),",
                f"score: {scores[idx]:.4f}"
            )

    # Draw the bounding boxes
    plt.figure()
    plt.imshow(im)
    ax = plt.gca()
    draw_boxes(ax, boxes)
    plt.show()

@torch.no_grad()
def main(args):

    dataset = DataFactory(name='hicodet', partition=args.partition, data_root=args.data_root)

    detector = build_detector(args, dataset.dataset.object_to_verb)
    detector.eval()

    if os.path.exists(args.resume):
        print(f"=> Continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        detector.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Start from a randomly initialised model")
    
    image, _ = dataset[args.index]
    output = detector([image])
    output[0]['original_size'] = dataset.dataset.image_size(args.index)
    visualise_entire_image(dataset, output[0])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-transformer', default=1e-5, type=float)
    parser.add_argument('--lr-backbone', default=1e-6, type=float)
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=15, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--num-classes', type=int, default=117)
    parser.add_argument('--human-idx', type=int, default=0)

    parser.add_argument('--device', default='cpu')
    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-h', default=3, type=int)
    parser.add_argument('--min-o', default=3, type=int)
    parser.add_argument('--max-h', default=15, type=int)
    parser.add_argument('--max-o', default=15, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--index', default=0, type=int)
    
    args = parser.parse_args()

    args.resume = 'checkpoints/upt-r50-hicodet-b16.pt'

    main(args)

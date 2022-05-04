"""
Visualise detected human-object interactions in an image

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import pocket
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import DataFactory
from upt import build_detector

warnings.filterwarnings("ignore")

OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i+1), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()

def visualise_entire_image(image, output, actions, action=None, thresh=0.2):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct
    # Find the number of human and object instances
    nh = len(output['pairing'][0].unique()); no = len(boxes)

    scores = output['scores']
    objects = output['objects']
    pred = output['labels']
    # Visualise detected human-object pairs with attached scores
    if action is not None:
        keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
        bx_h, bx_o = boxes[output['pairing']].unbind(0)
        pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
        plt.imshow(image)
        plt.axis('off')

        for i in range(len(keep)):
            txt = plt.text(*bx_h[keep[i], :2], f"{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()
        plt.show()
        return

    pairing = output['pairing']
    coop_attn = output['attn_maps'][0]
    comp_attn = output['attn_maps'][1]

    # Visualise attention from the cooperative layer
    for i, attn_1 in enumerate(coop_attn):
        fig, axe = plt.subplots(2, 4)
        fig.suptitle(f"Attention in coop. layer {i}")
        axe = np.concatenate(axe)
        ticks = list(range(attn_1[0].shape[0]))
        labels = [v + 1 for v in ticks]
        for ax, attn in zip(axe, attn_1):
            im = ax.imshow(attn.squeeze().T, vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)

    x, y = torch.meshgrid(torch.arange(nh), torch.arange(no))
    x, y = torch.nonzero(x != y).unbind(1)
    pairs = [str((i.item() + 1, j.item() + 1)) for i, j in zip(x, y)]

    # Visualise attention from the competitive layer
    fig, axe = plt.subplots(2, 4)
    fig.suptitle("Attention in comp. layer")
    axe = np.concatenate(axe)
    ticks = list(range(len(pairs)))
    for ax, attn in zip(axe, comp_attn):
        im = ax.imshow(attn, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(pairs, rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels(pairs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

    # Print predicted actions and corresponding scores
    unique_actions = torch.unique(pred)
    for verb in unique_actions:
        print(f"\n=> Action: {actions[verb]}")
        sample_idx = torch.nonzero(pred == verb).squeeze(1)
        for idx in sample_idx:
            idxh, idxo = pairing[:, idx] + 1
            print(
                f"({idxh.item():<2}, {idxo.item():<2}),",
                f"score: {scores[idx]:.4f}, object: {OBJECTS[objects[idx]]}."
            )

    # Draw the bounding boxes
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    ax = plt.gca()
    draw_boxes(ax, boxes)
    plt.show()

@torch.no_grad()
def main(args):

    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root)
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(dataset.dataset.object_to_action.values())
    args.num_classes = 117 if args.dataset == 'hicodet' else 24
    actions = dataset.dataset.verbs if args.dataset == 'hicodet' else \
        dataset.dataset.actions

    upt = build_detector(args, conversion)
    upt.eval()

    if os.path.exists(args.resume):
        print(f"=> Continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Start from a randomly initialised model")

    if args.image_path is None:
        image, _ = dataset[args.index]
        output = upt([image])
        image = dataset.dataset.load_image(
            os.path.join(dataset.dataset._root,
                dataset.dataset.filename(args.index)
        ))
    else:
        image = dataset.dataset.load_image(args.image_path)
        image_tensor, _ = dataset.transforms(image, None)
        output = upt([image_tensor])

    visualise_entire_image(image, output[0], actions, args.action, args.action_score_thresh)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--human-idx', type=int, default=0)

    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--action', default=None, type=int,
        help="Index of the action class to visualise.")
    parser.add_argument('--action-score-thresh', default=0.2, type=float,
        help="Threshold on action classes.")
    parser.add_argument('--image-path', default=None, type=str,
        help="Path to an image file.")
    
    args = parser.parse_args()

    main(args)

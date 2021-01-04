"""
Visualise all human-object pairs of a specified
interaction class in one image

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import argparse
import scipy.io as sio

from PIL import Image, ImageDraw

import pocket

def visualise_and_cache(args):
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    dataset = pocket.data.HICODet(
        os.path.join(args.data_root, "hico_20160224_det/images/{}".format(args.partition)),
        os.path.join(args.data_root, "instances_{}.json".format(args.partition))
    )

    with open('../hicodet/coco80tohico80.json', 'r') as f:
        coco2hico = json.load(f)
    hico2coco = dict(zip(coco2hico.values(), coco2hico.keys()))

    obj_idx = dataset.class_corr[args.interaction][1]
    intra_idx = dataset.object_to_interaction[obj_idx].index(args.interaction)
    fname = 'detections_{}.mat'.format(hico2coco[obj_idx].zfill(2))

    # Load data
    all_boxes = sio.loadmat(os.path.join(args.dir, fname))["all_boxes"]
    boxes_in_image = all_boxes[intra_idx, args.image_idx]

    boxes_h = boxes_in_image[:, :4]
    boxes_o = boxes_in_image[:, 4:8]
    scores = boxes_in_image[:, -1]

    image = Image.open(os.path.join(dataset._root, dataset._filenames[args.image_idx]))
    # Visualise box pairs
    for bh, bo, s in zip(boxes_h, boxes_o, scores):
        image_ = image.copy()

        canvas = ImageDraw.Draw(image_)
        canvas.rectangle(bh.tolist(), outline="#007CFF", width=10)
        canvas.rectangle(bo.tolist(), outline="#46FF00", width=10)
        b_h_centre = (bh[:2] + bh[2:]) / 2
        b_o_centre = (bo[:2] + bo[2:]) / 2
        canvas.line(
            b_h_centre.tolist() + b_o_centre.tolist(),
            fill='#FF4444', width=10
        )
        canvas.ellipse(
            (b_h_centre - 15).tolist() + (b_h_centre + 15).tolist(),
            fill='#FF4444'
        )
        canvas.ellipse(
            (b_o_centre - 15).tolist() + (b_o_centre + 15).tolist(),
            fill='#FF4444'
        )

        image_.save(os.path.join(args.cache_dir, '{}.png'.format(s)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Parse .mat files and visualise the box pairs")
    parser.add_argument("--dir",
                        help="Directory where .mat files are stored",
                        type=str, required=True)
    parser.add_argument("--image-idx",
                        help="Index of the image to be visualised",
                        type=int, required=True)
    parser.add_argument("--interaction",
                        help="Index of the interaction to be visualised",
                        type=int, required=True)
    parser.add_argument("--data-root",
                        help="Path to HICODet root directory",
                        type=str, default='../hicodet')
    parser.add_argument("--cache-dir",
                        help="Directory where visualisations will be saved",
                        type=str, default='./cache')
    parser.add_argument("--partition",
                        help="Choose between train2015 and test2015",
                        type=str, default="test2015")
    args = parser.parse_args()

    visualise_and_cache(args)

"""
Parse .mat files and visualise the box pairs

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from torchvision.ops.boxes import box_iou

import pocket

COCO2HICO = [
    49, 9, 18, 44, 0, 16, 73, 74, 11, 72, 31, 63, 48, 8, 10, 20, 28, 37, 56, 25, 30, 
    6, 79, 34, 2, 76, 36, 68, 64, 33, 59, 60, 62, 40, 4, 5, 58, 65, 67, 13, 78, 26, 
    32, 41, 61, 14, 3, 1, 54, 46, 15, 19, 38, 50, 29, 17, 22, 24, 51, 7, 27, 70, 75, 
    42, 45, 53, 39, 21, 43, 47, 69, 57, 52, 12, 23, 77, 55, 66, 35, 71
]
MIN_IOU = 0.5
DEBUG = False

def plot_pr_curve(scores, labels, num_gt, cache_dir):
    """Plot the p-r curve for a class"""
    scores = torch.as_tensor(scores)
    labels = torch.as_tensor(labels)

    prec, rec = pocket.utils.DetectionAPMeter.compute_pr_for_each(
        scores, labels, num_gt)
    ap = pocket.utils.AveragePrecisionMeter.compute_per_class_ap_with_11_point_interpolation(
        (prec.float(), rec.float())
    )

    plt.plot(rec.numpy(), prec.numpy())
    plt.savefig(os.path.join(cache_dir, "pr_ap={:.4f}.png".format(ap)))
    plt.close()

def plot_ranked_scores(scores, labels, cache_dir):
    """Plot the ranked scores for positive and negative examples"""
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    pos_idx = np.where(labels)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_order = np.argsort(scores[pos_idx])
    neg_order = np.argsort(scores[neg_idx])

    plt.figure(figsize=(10, 2))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(scores[pos_idx][pos_order], label="Scores")
    ax1.set_title("Positive examples")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(scores[neg_idx][neg_order], label="Scores")
    ax2.set_title("Negative examples")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid()

    plt.savefig(os.path.join(cache_dir, "ranked_scores.png"))
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Parse .mat files and visualise the box pairs")
    parser.add_argument("--dir",
                        help="Directory where .mat files are stored",
                        type=str, required=True)
    parser.add_argument("--data-root",
                        help="Path to HICODet root directory",
                        type=str, required=True)
    parser.add_argument("--partition",
                        help="Choose between train2015 and test2015",
                        type=str, default="test2015")
    parser.add_argument("--cache-dir",
                        help="Directory where visualisations will be saved",
                        type=str, required=True)
    parser.add_argument("--num-pos",
                        help="Maximum number of positive examples to visualise",
                        default=100, type=int)
    parser.add_argument("--num-neg",
                        help="Maximum number of negative examples to visualise",
                        default=100, type=int)
    args = parser.parse_args()

    dataset = pocket.data.HICODet(
        os.path.join(args.data_root, "hico_20160224_det/images/{}".format(args.partition)),
        os.path.join(args.data_root, "instances_{}.json".format(args.partition))
    )

    fnames = os.listdir(args.dir)
    # if DEBUG:
    #     fnames = [fnames[0]]
    for f in fnames:
        obj_idx = COCO2HICO[int(f[11:13]) - 1]
        all_boxes = sio.loadmat(os.path.join(args.dir, f))["all_boxes"]

        for intra_idx, boxes in enumerate(all_boxes):
            hoi_idx = dataset.object_to_interaction[obj_idx][intra_idx]

            # Extract the total number of detections
            num = 0
            for b in boxes:
                num += b.shape[0]

            labels = np.zeros(num); scores = np.zeros(num)
            box_idx = np.ones([num, 2]); counter = 0
            # Associate detections with ground truth
            for img_idx, boxes_in_image in enumerate(boxes):
                n = boxes_in_image.shape[0]
                # Skip images without detections
                if n == 0:
                    continue
                box_idx[counter: counter + n, 0] = img_idx
                box_idx[counter: counter + n, 1] = np.arange(n)
                scores[counter: counter + n] = boxes_in_image[:, -1]
                # Image does not contain valid interactions
                if img_idx in dataset._empty_idx:
                    counter += n
                    continue
                gt = dataset._anno[img_idx]
                # Current interaction not present in the image
                if hoi_idx not in gt["hoi"]:
                    counter += n
                    continue
                gt_idx = np.where(np.asarray(gt["hoi"]) == hoi_idx)[0]
                gt_bh = np.asarray(gt["boxes_h"])[gt_idx]
                gt_bo = np.asarray(gt["boxes_o"])[gt_idx]
                # Match detections with ground truth
                match = torch.min(
                    box_iou(
                        torch.from_numpy(gt_bh).float(),
                        torch.as_tensor(boxes_in_image[:, :4])),
                    box_iou(
                        torch.from_numpy(gt_bo).float(),
                        torch.as_tensor(boxes_in_image[:, 4:8])
                )) > MIN_IOU

                for i, m in enumerate(match):
                    match_idx = torch.nonzero(m).squeeze(1)
                    if len(match_idx) == 0:
                        continue
                    match_scores = boxes_in_image[match_idx, -1]
                    labels[counter + match_idx[match_scores.argmax()].item()] = 1

                # Increment counter
                counter += n

            cache_dir = os.path.join(args.cache_dir, "class_{:03d}".format(hoi_idx))
            example_cache_dir = os.path.join(cache_dir, "examples")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                os.mkdir(example_cache_dir)

            plot_pr_curve(scores, labels, dataset.anno_interaction[hoi_idx], cache_dir)
            if DEBUG:
                continue
            plot_ranked_scores(scores, labels, cache_dir)

            order = scores.argsort()[::-1]
            sorted_labels = labels[order]
            # Keep the top-k positive and negative examples
            pos_keep = np.where(sorted_labels)[0][:args.num_pos]
            neg_keep = np.where(sorted_labels==0)[0][:args.num_neg]

            keep = np.hstack([pos_keep, neg_keep])
            sorted_idx = box_idx[order][keep, :].astype(np.int64)

            for j, k in zip(keep, sorted_idx):
                image = Image.open(os.path.join(dataset._root, dataset._filenames[k[0]]))
                target = dataset._anno[k[0]]

                b1 = boxes[k[0]][k[1], :4]; b2 = boxes[k[0]][k[1], 4:8]

                canvas = ImageDraw.Draw(image)
                canvas.rectangle(b1.tolist(), outline="#007CFF", width=2)
                canvas.rectangle(b2.tolist(), outline="#46FF00", width=2)
                b_h_centre = (b1[:2]+b1[2:])/2
                b_o_centre = (b2[:2]+b2[2:])/2
                canvas.line(
                    b_h_centre.tolist() + b_o_centre.tolist(),
                    fill='#FF4444', width=2
                )
                canvas.ellipse(
                    (b_h_centre - 3).tolist() + (b_h_centre + 3).tolist(),
                    fill='#FF4444'
                )
                canvas.ellipse(
                    (b_o_centre - 3).tolist() + (b_o_centre + 3).tolist(),
                    fill='#FF4444'
                )

                # Get ground truth box pairs of the same class
                target = pocket.ops.to_tensor(target, input_format="dict")
                gt_idx = torch.nonzero(target["hoi"] == torch.as_tensor(hoi_idx)).squeeze(1)
                if len(gt_idx):
                    gt_b1 = target["boxes_h"][gt_idx]
                    gt_b2 = target["boxes_o"][gt_idx]
                    # Find the G.T. with highest IoU
                    idx = torch.min(
                        box_iou(torch.as_tensor(b1)[None, :].float(), gt_b1),
                        box_iou(torch.as_tensor(b2)[None, :].float(), gt_b2)
                    ).squeeze().argmax()

                    pocket.utils.draw_dashed_rectangle(
                        image, gt_b1[idx].tolist(),
                        fill="#007CFF", width=2
                    )
                    pocket.utils.draw_dashed_rectangle(
                        image, gt_b2[idx].tolist(),
                        fill="#46FF00", width=2
                    )

                # The image name is in the following format:
                # {RANK_{IMAGE_IDX}_{BOX_PAIR_IDX}_{LABEL}_{SCORE}.png
                cache_name = "{}_{}_{}_{}_{:.4f}.png".format(
                    j, *k, int(sorted_labels[j]),
                    scores[order][j]
                )
                image.save(os.path.join(example_cache_dir, cache_name))

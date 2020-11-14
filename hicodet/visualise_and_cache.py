"""
Visualise ground truth human-object pairs
and save as .png images

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import argparse
import numpy as np

from PIL import ImageDraw

import pocket

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visualise and cache human-object pairs")
    parser.add_argument("--partition",
                        help="Choose between train2015 and test2015",
                        required=True,
                        type=str)

    args = parser.parse_args()

    # Load dataset
    dataset = pocket.data.HICODet(
        "../hicodet/hico_20160224_det/images/{}".format(args.partition),
        "../hicodet/instances_{}.json".format(args.partition)
    )

    root_cache = os.path.join("./visualisations", args.partition)

    for idx, (image, target) in enumerate(dataset):
        classes = np.asarray(target["hoi"])
        unique_cls = np.unique(classes)
        # Visualise by class
        for cls_idx in unique_cls:
            sample_idx = np.where(classes == cls_idx)[0]
            image_ = image.copy()
            canvas = ImageDraw.Draw(image_)
            for i in sample_idx:
                b1 = target["boxes_h"][i]
                b2 = target["boxes_o"][i]

                canvas.rectangle(b1, outline='#007CFF', width=2)
                canvas.rectangle(b2, outline='#46FF00', width=2)
                canvas.line(
                    [(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2,
                    (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2],
                    fill='#FF4444', width=2
                )
            cache_dir = os.path.join(root_cache, "class_{:03d}".format(cls_idx))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            image_.save(os.path.join(
                cache_dir, "{}.png".format(idx)
            ))

        if idx % 500 == 0:
            print(idx)
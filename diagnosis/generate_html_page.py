"""
Generate code for HTML table to visualise human-object pairs

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import argparse
import numpy as np

import pocket

def name_parser(name):
    """Parse image names that are in the following format:

        {RANK}_{IMAGE_IDX}_{BOX_PAIR_IDX}_{LABEL}_{SCORE}.png
    """
    seg = name.split("_")
    sample_type = "Positive" if int(seg[3]) else "Negative"

    return "Rank: {} ".format(seg[0]) + sample_type \
        + "<br>Image: {}, Pair: {}<br>".format(seg[1], seg[2]) \
        + "Score: {}".format(seg[4][:-4])

def sorter(names):
    rank = []
    for n in names:
        rank.append(int(n.split("_")[0]))
    order = np.asarray(rank).argsort()
    return order

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate HTML table")
    parser.add_argument("--image-dir",
                        required=True,
                        type=str)

    args = parser.parse_args()

    table = pocket.utils.ImageHTMLTable(
        4, args.image_dir,
        parser=name_parser, sorter=sorter,
        width="75%"
    )

    table()

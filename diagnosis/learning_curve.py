"""
Plot learning curves

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

COLOURS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2'
]

def plot_learning_curve(source):
    if len(source) % 2 != 0:
        raise AssertionError("There should be an even number of arguments")
    n = int(len(source) / 2)
    labels = dict()
    for i in range(n):
        path = source[2 * i]
        description = source[2 * i + 1]
        labels[path] = description

    data = dict()
    colours = COLOURS[:n]
    for k in labels:
        f = open(k)
        ap = []
        line = f.readline()
        while line:
            if line[:6] == "Epoch:":
                seg = line.split()
                ap.append([float(seg[5][:-1]), float(seg[11][:-1])])
            line = f.readline()
        data[k] = np.asarray(ap)

    for k, c in zip(data, colours):
        plt.plot(data[k][:, 0], color=c, linewidth=1, label='{} train'.format(labels[k]))
        plt.plot(data[k][:, 1], color=c, linewidth=2, label='{} test'.format(labels[k]))
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    leg = plt.legend()
    leg.set_draggable(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
        nargs='+', type=str, required=True,
        help="Paths of log files and descriptions in the format PATH_1, DESC_1, ..."
    )

    args = parser.parse_args()
    plot_learning_curve(args.source)

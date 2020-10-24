import os
import numpy as np
import matplotlib.pyplot as plt

labels = {
    "cuda2": "seperate spatial graph",
    "cuda4": "spatial features concat.",
    "cuda5": "baseline w/ softmax",
    "cuda6": "spaital attention",
    "cuda7": "baseline"
}
colours = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
#    '#8c564b'
]

directory = "../log"

data = dict()

for k in labels:
    path = os.path.join(directory, k)
    f = open(path)
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
plt.legend()
plt.show()


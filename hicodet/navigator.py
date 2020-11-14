"""
HICODet dataset navigator

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import argparse
import numpy as np
from math import ceil
from PIL import Image, ImageDraw

from pocket.data import HICODet, DatasetTree

help_msg = """
****************************************
* Welcome to HICODet Dataset Navigator *
****************************************
\nCommands are listed below:\n
path(p) - Print path of the current node
list(l) - List all navigable nodes
move(m) - Move to a navigable node
help(h) - Print help manual
exit(e) - Terminate the program
"""

def parse_commands(line):
    """Parse a line into commands and arguments"""
    segments = line.split()
    if len(segments) == 1:
        return segments[0], None
    elif len(segments) > 1:
        return segments[0], segments[1]
    else:
        return None, None

def list_node(tree, dataset):
    if tree.cn().name == "root":
        print("\t\t".join(tree.ls()))
    elif tree.cn().name == "images":
        pool = ["[{}] {}".format(k, sum(list(v.data.values()))
            ).ljust(20) for k, v in tree.cn().children.items()]
        for i in range(ceil(len(pool) / 4)):
            print("".join(pool[4*i:4*i+4]) + "\n")
    elif tree.cn().name == "classes":
        print("\n".join([
            "[{:>3}] {:>30}\t({})".format(
                k, dataset.interactions[int(k)], 
                sum(list(v.data.values()))
            ) for k, v in tree.cn().children.items()
        ]))
    elif tree.cn().name.isdigit():
        pool = ["[{}] {}".format(k, v).ljust(20)
            for k, v in tree.cn().data.items()]
        for i in range(ceil(len(pool) / 4)):
            print("".join(pool[4*i: 4*i+4]) + "\n")
    else:
        raise NotImplementedError("Unable to handle current path")

def visualise(dataset, image_idx, class_idx):
    """Visualise all box pairs of the same class in an image"""
    image, target = dataset[image_idx]
    canvas = ImageDraw.Draw(image)

    box_pair_idx = np.where(np.asarray(target["hoi"])==class_idx)[0]
    boxes_h = np.asarray(target["boxes_h"])[box_pair_idx]
    boxes_o = np.asarray(target["boxes_o"])[box_pair_idx]
    for b_h, b_o in zip(boxes_h, boxes_o):
        canvas.rectangle(b_h.tolist(), outline='#007CFF', width=2)
        canvas.rectangle(b_o.tolist(), outline='#46FF00', width=2)
        canvas.line((
            (b_h[:2]+b_h[2:])/2).tolist() + ((b_o[:2]+b_o[2:])/2).tolist(),
            fill='#FF4444', width=2
        )
    image.show()

def move(tree, dataset, args):
    dest = args.pop(0)
    if dest == "..":
        tree.up()
    elif dest in tree.cn().children:
        tree.down(dest)
    elif dest in tree.cn().data:
        idx1 = int(dest); idx2 = int(tree.path().split("/")[2])
        if tree.cn().parent.name == "images":
            visualise(dataset, idx2, idx1)
        else:
            visualise(dataset, idx1, idx2)
    else:
        print("Unknown destination \"{}\"".format(dest))
    # Recursively move to the desitination
    if len(args):
        move(tree, dataset, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("HICODet dataset navigator")
    parser.add_argument("--partition",
                        required=True,
                        type=str)

    args = parser.parse_args()
    
    dataset = HICODet(
        root="./hico_20160224_det/images/{}".format(args.partition),
        annoFile="./instances_{}.json".format(args.partition),
    )

    image_labels = [dataset.annotations[i]["hoi"] for i in dataset._idx]
    tree = DatasetTree(600, image_labels)

    print(help_msg)
    while(1):
        try:
            line = input("> ").lower()
        except EOFError:
            exit()
        
        cmd, args = parse_commands(line)

        if cmd is None:
            continue
        elif cmd in ["path", "p"]:
            print(tree.path())
        elif cmd in ["list", "l"]:
            list_node(tree, dataset)
        elif cmd in ["move", "m"]:
            move(tree, dataset, args.split("/"))
        elif cmd in ["help", "h"]:
            print(help_msg)
        elif cmd in ["exit", "e"]:
            exit()
        else:
            print("Unknown command \"{}\"".format(cmd))
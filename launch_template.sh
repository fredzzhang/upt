#!/bin/bash

# This is a template for launching training and testing scripts
# 
# Fred Zhang <frederic.zhang@anu.edu.au>
# 
# The Australian National University
# Australian Centre for Robotic Vision

# -------------------------------
# Training commands
# -------------------------------

# Train UPT-R50 on HICO-DET
python main.py --world-size 8 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/upt-r50-hicodet
# Train UPT-R101 on HICO-DET
python main.py --world-size 8 --backbone resnet101 --pretrained checkpoints/detr-r101-hicodet.pth --output-dir checkpoints/upt-r101-hicodet
# Train UPT-R101-DC5 on HICO-DET
python main.py --world-size 8 --backbone resnet101 --dilation --pretrained checkpoints/detr-r101-dc5-hicodet.pth --output-dir checkpoints/upt-r101-dc5-hicodet

# Train UPT-R50 on V-COCO
python main.py --world-size 8 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/upt-r50-vcoco
# Train UPT-R101 on V-COCO
python main.py --world-size 8 --dataset vcoco --data-root vcoco/ --partitions trainval test --backbone resnet101 --pretrained checkpoints/detr-r101-vcoco.pth --output-dir checkpoints/upt-r101-vcoco
# Train UPT-R101-DC5 on V-COCO
python main.py --world-size 8 --dataset vcoco --data-root vcoco/ --partitions trainval test --backbone resnet101 --dilation --pretrained checkpoints/detr-r101-dc5-vcoco.pth --output-dir checkpoints/upt-r101-dc5-vcoco

# -------------------------------
# Testing commands
# -------------------------------

# Test UPT-R50 on HICO-DET
python main.py --eval --resume /path/to/model
# Test UPT-R101 on HICO-DET
python main.py --eval --backbone resnet101 --resume /path/to/model
# Test UPT-R101-DC5 on HICO-DET
python main.py --eval --backbone resnet101 --dilation --resume /path/to/model

# Cache detection results from UPT-R50 for Matlab evaluation on HICO-DET
python main.py --cache --output-dir matlab-r50 --resume /path/to/model
# Cache detection results from UPT-R101 for Matlab evaluation on HICO-DET
python main.py --cache --backbone resnet101 --output-dir matlab-r101 --resume /path/to/model
# Cache detection results from UPT-R101-DC5 for Matlab evaluation on HICO-DET
python main.py --cache --backbone resnet101 --dilation --output-dir matlab-r101-dc5 --resume /path/to/model

# Cache detection results from UPT-R50 for evaluation on V-COCO
python main.py --cache --dataset vcoco --data-root vcoco/ --partitions trainval test --output-dir vcoco-r50 --resume /path/to/model
# Cache detection results from UPT-R101 for evaluation on V-COCO
python main.py --cache --dataset vcoco --data-root vcoco/ --partitions trainval test --backbone resnet101 --output-dir vcoco-r101 --resume /path/to/model
# Cache detection results from UPT-R101-DC5 for evaluation on V-COCO
python main.py --cache --dataset vcoco --data-root vcoco/ --partitions trainval test --backbone resnet101 --dilation --output-dir vcoco-r101-dc5 --resume /path/to/model
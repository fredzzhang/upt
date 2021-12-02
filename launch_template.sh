# Train on HICO-DET with ResNet50
python main.py --world-size 8 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/upt-r50-hicodet

# Train on HICO-DET with ResNet101
python main.py --world-size 8 --backbone resnet101  --pretrained checkpoints/detr-r101-hicodet.pth --output-dir checkpoints/upt-r101-hicodet

# Train on HICO-DET with ResNet101-DC5
python main.py --world-size 8 --backbone resnet101 --dilation --pretrained checkpoints/detr-r101-dc5-hicodet.pth --output-dir checkpoints/upt-r101-dc5-hicodet

# Train on V-COCO with ResNet50
python main.py --world-size 8 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/upt-r50-vcoco

# Train on V-COCO with ResNet101
python main.py --world-size 8 --dataset vcoco --data-root vcoco/ --partitions trainval test --backbone resnet101 --pretrained checkpoints/detr-r101-vcoco.pth --output-dir checkpoints/upt-r101-vcoco

# Train on V-COCO with ResNet101-DC5
python main.py --world-size 8 --dataset vcoco --data-root vcoco/ --partitions trainval test --backbone resnet101 --dilation --pretrained checkpoints/detr-r101-dc5-vcoco.pth --output-dir checkpoints/upt-r101-dc5-vcoco
import os
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader

import pocket
from pocket.data import HICODet

from models import InteractGraphNet
from utils import CustomisedDataset, custom_collate, test

def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    testset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/test2015"),
        annoFile=os.path.join(args.data_root,
            "instances_test2015.json"),
        transform=torchvision.transforms.ToTensor(),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )    
    test_loader = DataLoader(
        dataset=CustomisedDataset(testset,
            os.path.join(args.data_root,
            "fasterrcnn_resnet50_fpn_detections/test2015"),
            human_idx=49
        ), collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )

    net = InteractGraphNet(
        testset.object_to_verb, 49,
        num_iterations=args.num_iter
    )
    epoch = 0
    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint["epoch"]

    net.cuda()
    timer = pocket.utils.HandyTimer(maxlen=1)
    
    with timer:
        test_ap = test(net, test_loader)
    print("Epoch: {} | test mAP: {:.4f}, total time: {:.2f}s".format(
        epoch, test_ap.mean(), timer[0]
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', required=True, type=str)
    parser.add_argument('--num-iter', default=1, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--batch-size', default=2, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='', type=str)
    
    args = parser.parse_args()
    print(args)

    main(args)
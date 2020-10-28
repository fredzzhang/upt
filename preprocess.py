import os
import torch
import pickle
import argparse
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

import pocket
from pocket.data import HICODet

from models import InteractGraphNet
from utils import custom_collate, CustomisedDataset

def preprocess(net, dataloader, training, cache_path):
    all_results = []
    for idx, batch in enumerate(tqdm(dataloader)):
        batch_cuda = pocket.ops.relocate_to_cuda(batch)
        images, detections, targets = batch_cuda
        # Preprocess
        images, detections, targets, original_image_sizes = net.preprocess(
            images, detections, targets
        )
        # Run backbone CNN
        features = net.backbone(images.tensors)
        # Run interaction head
        detections = net.interaction_head.preprocess(
            detections, targets, append_gt=training
        )
        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        box_scores = [detection['scores'] for detection in detections]
        box_features = net.interaction_head.box_roi_pool(
            features, box_coords, images.image_sizes
        )
        # Run box pair head
        box_pair_features, boxes_h, boxes_o, object_class,\
        box_pair_labels, box_pair_prior = net.interaction_head.box_pair_head(
            features, images.image_sizes, box_features,
            box_coords, box_labels, box_scores, targets
        )
        results = []
        for b_idx, (f, b_h, b_o, obj, l, p) in enumerate(zip(
            box_pair_features, boxes_h, boxes_o, object_class,
            box_pair_labels, box_pair_prior
        )):
            # Skip images without valid box pairs
            if len(f) == 0:
                continue
            results.append(dict(
                features=f, boxes_h=b_h, boxes_o=b_o,
                object_class=obj, labels=l, prior=p,
                index=idx * dataloader.batch_size + b_idx
            ))
        results = net.transform.postprocess(
            results, images.image_sizes,
            original_image_sizes
        )
        all_results += results
    with open(cache_path, 'wb') as f:
        pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)

def main(args):

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    trainset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/train2015"),
        annoFile=os.path.join(args.data_root,
            "instances_train2015.json"),
        transform=torchvision.transforms.ToTensor(),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )

    testset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/test2015"),
        annoFile=os.path.join(args.data_root,
            "instances_test2015.json"),
        transform=torchvision.transforms.ToTensor(),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )

    train_loader = DataLoader(
            dataset=CustomisedDataset(trainset, 
                os.path.join(args.data_root,
                "fasterrcnn_resnet50_fpn_detections/train2015"),
                human_idx=49,
                box_score_thresh_h=args.human_thresh,
                box_score_thresh_o=args.object_thresh
            ), collate_fn=custom_collate, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True
    )

    test_loader = DataLoader(
            dataset=CustomisedDataset(testset,
                os.path.join(args.data_root,
                "fasterrcnn_resnet50_fpn_detections/test2015"),
                human_idx=49,
                box_score_thresh_h=args.human_thresh,
                box_score_thresh_o=args.object_thresh
            ), collate_fn=custom_collate, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True
    )


    # Fix random seed for model synchronisation
    torch.manual_seed(args.random_seed)

    net = InteractGraphNet(
        trainset.object_to_verb, 49,
        num_iterations=args.num_iter
    )

    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        net.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    net.cuda()
    net.eval()

    preprocess(net, train_loader, True, os.path.join(args.cache_dir, "train2015.pkl"))
    preprocess(net, test_loader, False, os.path.join(args.cache_dir, "test2015.pkl"))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', required=True, type=str)
    parser.add_argument('--model-path', default='', type=str, required=True)
    parser.add_argument('--num-iter', default=1, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--human-thresh', default=0.5, type=float)
    parser.add_argument('--object-thresh', default=0.5, type=float)
    parser.add_argument('--batch-size', default=2, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    print(args)

    main(args)

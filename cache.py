"""
Run inference and cache detected box pairs and predictions

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

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
from utils import CustomisedDataset, custom_collate

def inference(net, dataloader, batch_size):
    dataset = dataloader.dataset.dataset
    net.eval()
    all_results = dict()
    for i, batch in enumerate(tqdm(dataloader)):
        inputs = pocket.ops.relocate_to_cuda(batch[:-1])
        with torch.no_grad():
            output = net(*inputs)
        if output is None:
            continue

        output = pocket.ops.relocate_to_cpu(output)
        # Organise output results
        for j, result in enumerate(output):
            filename = dataset.filename(i*batch_size+j).split('.')[0]

            box_idx = result['index']
            boxes_h = result['boxes_h'][box_idx]
            boxes_o = result['boxes_o'][box_idx]
            objects = result['object'][box_idx]

            scores = result['scores']
            verbs = result['prediction']
            interactions = torch.tensor([
                dataset.object_n_verb_to_interaction[o][v]
                for o, v in zip(objects, verbs)
            ])

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Record start and end index of box pairs for each class
            idx = torch.zeros(600, 2)
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                idx[cls_id.long(), 0] = n
                idx[cls_id.long(), 1] = n + cls_num
                n += cls_num

            all_results[filename] = dict(
                start_end_ids=idx.numpy(),
                human_obj_boxes_scores=torch.cat([
                    boxes_h, boxes_o, scores[:, None]
                ], 1)
            )
    # Cache results
    with open('test.pkl', 'wb') as f:
        pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)

def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    dataset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/{}".format(args.partition)),
        annoFile=os.path.join(args.data_root,
            "instances_{}.json".format(args.partition)),
        transform=torchvision.transforms.ToTensor(),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )    
    dataloader = DataLoader(
        dataset=CustomisedDataset(dataset,
            os.path.join(args.data_root,
            "fasterrcnn_resnet50_fpn_detections/{}".format(args.parition)),
            human_idx=49
        ), collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )

    net = InteractGraphNet(
        dataset.object_to_verb, 49,
        num_iterations=args.num_iter
    )
    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])

    net.cuda()
    
    inference(net, dataloader, args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', required=True, type=str)
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=3, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--batch-size', default=2, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='', type=str)
    
    args = parser.parse_args()
    print(args)

    main(args)

import os
import time
import torch
import argparse
import torchvision
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.ops.boxes import box_iou

import pocket
from pocket.data import HICODet
from pocket.utils import NumericalMeter, DetectionAPMeter, HandyTimer

from models import InteractGraphNet
from utils import custom_collate, CustomisedEngine, CustomisedDataset

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
                human_idx=49
            ), collate_fn=custom_collate, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True
    )

    test_loader = DataLoader(
            dataset=CustomisedDataset(testset,
                os.path.join(args.data_root,
                "fasterrcnn_resnet50_fpn_detections/test2015"),
                human_idx=49
            ), collate_fn=custom_collate, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True
    )


    # Fix random seed for model synchronisation
    torch.manual_seed(args.random_seed)

    net = InteractGraphNet(
        trainset.object_to_verb, 49,
        num_iterations=args.num_iter
    )
    # Fix backbone parameters
    for p in net.backbone.parameters():
        p.requires_grad = False

    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        net.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    if not os.path.exists(args.cache_dir):
        os.mkdir(args.cache_dir)

    net.cuda()

    net_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(net_params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=args.milestones,
        gamma=args.lr_decay
    )

    running_loss = NumericalMeter(maxlen=args.print_interval)
    t_data = NumericalMeter(maxlen=args.print_interval)
    t_iteration = NumericalMeter(maxlen=args.print_interval)
    timer = HandyTimer(2)

    iterations = 0

    for epoch in range(args.num_epochs):
        #################
        # on_start_epoch
        #################
        net.train()
        ap_train = DetectionAPMeter(117, algorithm='INT')
        timestamp = time.time()
        for batch in train_loader:
            ####################
            # on_start_iteration
            ####################
            iterations += 1
            batch_cuda = pocket.ops.relocate_to_cuda(batch)
            t_data.append(time.time() - timestamp)
            ####################
            # on_each_iteration
            ####################
            optimizer.zero_grad()
            output = net(*batch_cuda)
            loss = output.pop()
            loss.backward()
            optimizer.step()

            if output is None:
                continue

            # Collate results within the batch
            for result_one in output:
                ap_train.append(
                    torch.cat(result_one["scores"]),
                    torch.cat(result_one["labels"]),
                    torch.cat(result_one["gt_labels"])
                )
            ####################
            # on_end_iteration
            ####################
            running_loss.append(loss.item())
            t_iteration.append(time.time() - timestamp)
            timestamp = time.time()
            if iterations % args.print_interval == 0:
                avg_loss = running_loss.mean()
                sum_t_data = t_data.sum()
                sum_t_iter = t_iteration.sum()
                
                num_iter = len(train_loader)
                n_d = len(str(num_iter))
                print(
                    "Epoch [{}/{}], Iter. [{}/{}], "
                    "Loss: {:.4f}, "
                    "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
                    epoch+1, args.num_epochs,
                    str(iterations - num_iter * epoch).zfill(n_d),
                    num_iter, avg_loss, sum_t_data, sum_t_iter
                ))
                running_loss.reset()
                t_data.reset(); t_iteration.reset()
        #################
        # on_end_epoch
        #################
        lr_scheduler.step()
        torch.save({
            'iteration': iterations,
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optim_state_dict': optimizer.state_dict()
            }, os.path.join(args.cache_dir, 'ckpt_{:05d}_{:02d}.pt'.\
                    format(iterations, epoch+1)))
        
        ap_test = DetectionAPMeter(117, num_gt=test_loader.dataset.anno_action, algorithm='11P')
        net.eval()
        for batch in test_loader:
            inputs = pocket.ops.relocate_to_cuda(batch[:-1])
            with torch.no_grad():
                output = net(*inputs)
            if output is None:
                continue

            for result, target in zip(output, batch[-1]):
                result = pocket.ops.relocate_to_cpu(result)

                # Reformat the predicted classes and scores
                # CLASS_IDX: [SCORE, BINARY_LABELS]
                predictions = [{
                    int(k.item()):[v.item(), 0]
                    for k, v in zip(l, s)
                } for l, s in zip(result["labels"], result["scores"])]
                # Compute minimum IoU
                iou = torch.min(
                    box_iou(target['boxes_h'], result['boxes_h']),
                    box_iou(target['boxes_o'], result['boxes_o'])
                )
                # Assign each detection to GT with the highest IoU
                max_iou, max_idx = iou.max(0)
                match = -1 * torch.ones_like(iou)
                match[max_idx, torch.arange(iou.shape[1], device=iou.device)] = max_iou
                match = match >= 0.5

                # Associate detected box pairs with ground truth
                for i, m in enumerate(match):
                    target_class = target["labels"][i].item()
                    # For each ground truth box pair, find the 
                    # detections with sufficient IoU
                    det_idx = m.nonzero().squeeze(1)
                    if len(det_idx) == 0:
                        continue
                    # Retrieve the scores of matched detections
                    # When target class was not predicted, fill the score as -1
                    match_scores = torch.as_tensor([p[target_class][0]
                        if target_class in p else -1 for p in predictions])
                    match_idx = match_scores.argmax()
                    # None of matched detections have predicted target class
                    if match_scores[match_idx] == -1:
                        continue
                    predictions[match_idx][target_class][2] = 1

                pred = torch.cat([
                    torch.tensor(list(p.keys())) for p in predictions
                ])
                scores, labels = torch.cat([
                    torch.tensor(list(p.values())) for p in predictions
                ]).unbind(1)

                ap_test.append(scores, pred, labels)
        
        with timer:
            ap_1 = ap_train.eval()
        with timer:
            ap_2 = ap_test.eval()

        
        print("Epoch: {} | training mAP: {:.4f}, time: {:.2f}s |"
            "test mAP: {:.4f}, time: {:.2f}s".format(
                epoch+1, ap_1.mean().item(), timer[0],
                ap_2.mean().item(), timer[1]
            ))
        
        timer.reset()

            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', required=True, type=str)
    parser.add_argument('--num-iter', default=1, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--num-epochs', default=20, type=int)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--lr-decay', default=0.1, type=float,
                        help="The multiplier by which the learning rate is reduced")
    parser.add_argument('--milestones', nargs='+', default=[10, 15],
                        help="The epoch number when learning rate is reduced")
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--print-interval', default=100, type=int)
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    print(args)

    main(args)
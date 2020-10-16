import os
import time
import torch
import argparse
import torchvision
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import pocket
from pocket.data import HICODet
from pocket.utils import SyncedNumericalMeter, DetectionAPMeter, all_gather

from models import InteractGraphNet
from utils import custom_collate, CustomisedEngine, CustomisedDataset

def main(rank, args):

    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

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
            num_workers=args.num_workers, pin_memory=True,
            sampler=DistributedSampler(
                trainset, 
                num_replicas=args.world_size, 
                rank=rank)
    )

    test_loader = DataLoader(
            dataset=CustomisedDataset(testset,
                os.path.join(args.data_root,
                "fasterrcnn_resnet50_fpn_detections/test2015"),
                human_idx=49
            ), collate_fn=custom_collate, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
            sampler=DistributedSampler(
                testset,
                num_replicas=args.world_size,
                rank=rank)
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
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=args.milestones,
        gamma=args.lr_decay
    )

    running_loss = SyncedNumericalMeter(maxlen=args.print_interval)
    t_data = SyncedNumericalMeter(maxlen=args.print_interval)
    t_iteration = SyncedNumericalMeter(maxlen=args.print_interval)

    iterations = 0

    for i in range(args.num_epoch):
        #################
        # on_start_epoch
        #################
        train_loader.sampler.set_epoch(i)
        model.train()
        if rank == 0:
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
            output = model(*batch_cuda)
            loss = output.pop()
            loss.backward()
            optimizer.step()

            scores = []; pred = []; labels = []
            # Collate results within the batch
            for result_one in output:
                scores.append(torch.cat(result_one["scores"]).detach().cpu().numpy())
                pred.append(torch.cat(result_one["labels"]).detach().cpu().numpy())
                labels.append(torch.cat(result_one["gt_labels"]).detach().cpu().numpy())
            # Sync across subprocesses
            score_list = all_gather(np.concatenate(scores))
            pred_list = all_gather(np.concatenate(pred))
            label_list = all_gather(np.concatenate(labels))
            # Collate and log results in master process
            if rank == 0:
                scores = torch.from_numpy(np.concatenate(score_list))
                pred = torch.from_numpy(np.concatenate(pred_list))
                labels = torch.from_numpy(np.concatenate(label_list))
                ap_train.append(scores, pred, labels)
            ####################
            # on_end_iteration
            ####################
            running_loss.append(loss.item())
            t_iteration.append(time.time() - timestamp)
            timestamp = time.time()
            if iterations % args.print_interval == 0:
                avg_loss = running_loss.mean()
                sum_t_data = t_data.sum() / args.world_size
                sum_t_iter = t_iteration.sum() / args.world_size
                # Print stats in the master process
                if rank == 0:
                    num_iter = len(train_loader)
                    n_d = len(str(num_iter))
                    print(
                        "Epoch [{}/{}], Iter. [{}/{}], "
                        "Loss: {:.4f}, "
                        "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
                        i+1, args.num_epochs,
                        str(iterations - num_iter * i).zfill(n_d),
                        num_iter, avg_loss, sum_t_data, sum_t_iter
                    ))
                running_loss.reset()
                t_data.reset(); t_iteration.reset()
        #################
        # on_end_epoch
        #################
        lr_scheduler.step()
        if rank == 0:
            torch.save({
                'iteration': iterations,
                'epoch': i+1,
                'model_state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict()
                }, os.path.join(args.cache_dir, 'ckpt_{:05d}_{:02d}.pt'.\
                        format(iterations, i+1)))


            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', required=True, type=str)
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
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

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))
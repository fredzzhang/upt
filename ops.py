"""
Opearations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.distributed as dist
import torchvision.ops.boxes as box_ops

from torch import nn, Tensor
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment

import sys
sys.path.append('detr')
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):

    def __init__(self,
        cost_object: float = 1., cost_verb: float = 1.,
        cost_bbox: float = 1., cost_giou: float = 1.
    ) -> None:
        """
        Parameters:
        ----------
        cost_object: float
            Weight on the object classification term
        cost_verb: float
            Weight on the verb classification term
        cost_bbox:
            Weight on the L1 regression error
        cost_giou:
            Weight on the GIoU term
        """
        super().__init__()
        self.cost_object = cost_object
        self.cost_verb = cost_verb
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_object + cost_verb + cost_bbox + cost_giou, \
            "At least one cost coefficient should be non zero."

    @torch.no_grad()
    def forward(self,
        bx_h: List[Tensor], bx_o: List[Tensor], objects: List[Tensor],
        prior: List[Tensor], logits: Tensor, targets: List[dict]
    ) -> List[Tensor]:
        """
        Parameters:
        ----------
        bh: List[Tensor]
            (M, 4) Human bounding boxes in detected pairs
        bo: List[Tensor]
            (M, 4) Object bounding boxes in detected pairs
        objects: List[Tensor]
            (M,) Object class indices in each pair 
        prior: List[Tensor]
            (2, M, K) Object detection scores for the human and object boxes in each pair
        logits: Tensor
            (M_, K) Classification logits for all boxes pairs
        targets: List[dict]
            Targets for each image with the following keys, `boxes_h` (G, 4), `boxes_o` (G, 4),
            `labels` (G, 117), `objects` (G,)

        Returns:
        --------
        List[Tensor]
            A list of tuples for matched indices between detected pairs and ground truth pairs.

        """
        eps = 1e-6

        # The number of box pairs in each image
        n = [len(p) for p in bx_h]

        gt_bx_h = [t['boxes_h'] for t in targets]
        gt_bx_o = [t['boxes_o'] for t in targets]

        scores = [
            torch.sigmoid(lg) * p.prod(0)
            for lg, p in zip(logits.split(n), prior)
        ]
        gt_labels = [t['labels'] for t in targets]

        cost_verb = [
            (s.matmul(l.T) / (l.sum(dim=1, keepdim=True) + eps) +
            (1-s).matmul(1 - l.T) / (torch.sum(1 - l, dim=1, keepdim=True) + eps)) / 2
            for s, l in zip(scores, gt_labels)
        ]

        cost_bbox = [torch.max(
            torch.cdist(h, gt_h, p=1), torch.cdist(o, gt_o, p=1)
        ) for h, o, gt_h, gt_o in zip(bx_h, bx_o, gt_bx_h, gt_bx_o)]

        cost_giou = [torch.max(
            -generalized_box_iou(box_cxcywh_to_xyxy(h), box_cxcywh_to_xyxy(gt_h)),
            -generalized_box_iou(box_cxcywh_to_xyxy(o), box_cxcywh_to_xyxy(gt_o))
        ) for h, o, gt_h, gt_o in zip(bx_h, bx_o, gt_bx_h, gt_bx_o)]

        cost_object = [
            -torch.log(                                 # Log barrier
                obj.unsqueeze(1).eq(t['objects'])       # Binary mask
                * p[0].max(-1)[0].unsqueeze(1)          # Object classification score
            ) for obj, p, t in zip(objects, prior, targets)
        ]

        # Final cost matrix
        C = [
            c_v * self.cost_verb + c_b * self.cost_bbox +
            c_g * self.cost_giou + c_o * self.cost_object
            for c_v, c_b, c_g, c_o in zip(cost_verb, cost_bbox, cost_giou, cost_object)
        ]

        indices = [linear_sum_assignment(c.cpu()) for c in C]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, args) -> None:
        self.args = args
        self.matcher = HungarianMatcher(
            cost_object=args.cost_object,
            cost_verb=args.cost_verb,
            cost_bbox=args.cost_bbox,
            cost_giou=args.cost_giou
        )
        super().__init__()

    def interaction_loss(self,
        bx_h: List[Tensor], bx_o: List[Tensor], indices: List[Tensor],
        prior: List[Tensor], logits: Tensor, targets: List[dict]
    ) -> Tensor:
        collated_labels = []
        for bh, bo, idx, tgt in zip(bx_h, bx_o, indices, targets):
            idx_h, idx_o = idx

            mask = torch.diag(torch.min(
                box_ops.box_iou(bh[idx_h], tgt['boxes_h'][idx_o]),
                box_ops.box_iou(bo[idx_h], tgt['boxes_o'][idx_o])
            ) > 0.5).unsqueeze(1)
            matched_labels = tgt['labels'][idx_o] * mask
            labels = torch.zeros(
                len(bh), self.args.num_classes,
                device=matched_labels.device
            )
            labels[idx_h] = matched_labels
            collated_labels.append(labels)

        collated_labels = torch.cat(collated_labels)
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]; prior = prior[x, y]; labels = collated_labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                (prior + 1e-8) / (1 + torch.exp(-logits) - prior)
            ), labels, reduction='sum', alpha=self.args.alpha, gamma=self.args.gamma
        )

        return loss / n_p

    def forward(self,
        boxes: List[Tensor], bh: List[Tensor], bo: List[Tensor], objects: List[Tensor],
        prior: List[Tensor], logits: Tensor, targets: List[dict]
    ) -> Tensor:
        bx_h = [b[h] for b, h in zip(boxes, bh)]
        bx_o = [b[o] for b, o in zip(boxes, bo)]
        indices = self.matcher(bx_h, bx_o, objects, prior, logits, targets)

        return dict(
            interaction_loss=self.interaction_loss(bx_h, bx_o, indices, prior, logits, targets)
        )


def compute_spatial_encodings(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> Tensor:
    """
    Parameters:
    -----------
    boxes_1: List[Tensor]
        First set of bounding boxes (M, 4)
    boxes_1: List[Tensor]
        Second set of bounding boxes (M, 4)
    shapes: List[Tuple[int, int]]
        Image shapes, heights followed by widths
    eps: float
        A small constant used for numerical stability

    Returns:
    --------
    Tensor
        Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)

def binary_focal_loss_with_logits(
    x: Tensor, y: Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: float = 1e-6
) -> Tensor:
    """
    Focal loss by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf

    L = - |1-y-alpha| * |y-x|^{gamma} * log(|1-y-x|)

    Parameters:
    -----------
    x: Tensor[N, K]
        Post-normalisation scores
    y: Tensor[N, K]
        Binary labels
    alpha: float
        Hyper-parameter that balances between postive and negative examples
    gamma: float
        Hyper-paramter suppresses well-classified examples
    reduction: str
        Reduction methods
    eps: float
        A small constant to avoid NaN values from 'PowBackward'

    Returns:
    --------
    loss: Tensor
        Computed loss tensor
    """
    loss = (1 - y - alpha).abs() * ((y-torch.sigmoid(x)).abs() + eps) ** gamma * \
        torch.nn.functional.binary_cross_entropy_with_logits(
            x, y, reduction='none'
        )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction method {}".format(reduction))

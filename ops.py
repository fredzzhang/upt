"""
Opearations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torchvision.ops.boxes as box_ops

def compute_spatial_encodings(boxes_1, boxes_2, shapes, eps=1e-10):
    """
    Arguments:
        boxes_1[List[Tensor[M, 4]]]
        boxes_1[List[Tensor[M, 4]]]
        shapes(List[Tuple[height, width]])
        eps(float): A constant used for numerical stability
    Returns:
        Tensor[N, 36]
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

def binary_focal_loss(x, y, alpha=0.5, gamma=2.0, reduction='mean', eps=1e-6):
    """
    Focal loss by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf

    L = - |1-y-alpha| * |y-x|^{gamma} * log(|1-y-x|)

    Arguments:
        x(Tensor[N, K]): Post-normalisation scores
        y(Tensor[N, K]): Binary labels
        alpha(float): Hyper parameter
        gamma(float): Hyper paramter
        reduction(str): Reduction methods
        eps(float): A small constant to avoid NaN values from 'PowBackward'
    """
    loss = (1 - y - alpha).abs() * ((y-x).abs() + eps) ** gamma * \
        torch.nn.functional.binary_cross_entropy(
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

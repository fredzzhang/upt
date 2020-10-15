"""
Models

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torchvision.ops.boxes as box_ops

from torch import nn
from torchvision.ops._utils import _cat
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import transform

import pocket.models as models

def LIS(x, T=8.3, k=12, w=10):
    """
    Low-grade suppression
    https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network
    """
    return T / ( 1 + torch.exp(k - w * x)) 

class InteractGraph(nn.Module):
    def __init__(self,
                node_encoding_size, 
                representation_size, 
                num_cls, human_idx,
                object_class_to_target_class,
                fg_iou_thresh=0.5,
                num_iter=1):

        super().__init__()

        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size

        self.num_cls = num_cls
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter

        # Compute adjacency matrix
        self.adjacency = nn.Sequential(
            nn.Linear(node_encoding_size*2, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, int(representation_size/2)),
            nn.ReLU(),
            nn.Linear(int(representation_size/2), 1)
        )

        # Compute messages
        self.sub_to_obj = nn.Sequential(
            nn.Linear(node_encoding_size, representation_size),
            nn.ReLU()
        )
        self.obj_to_sub = nn.Sequential(
            nn.Linear(node_encoding_size, representation_size),
            nn.ReLU()
        )

        # Update node hidden states
        self.sub_update = nn.Linear(
            node_encoding_size + representation_size,
            node_encoding_size,
            bias=False
        )
        self.obj_update = nn.Linear(
            node_encoding_size + representation_size,
            node_encoding_size,
            bias=False
        )

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        """
        Arguements:
            boxes_h(Tensor[N, 4])
            boxes_o(Tensor[N, 4])
            targets(dict[Tensor]): Targets in an image with the following keys
                "boxes_h": Tensor[N, 4]
                "boxes_o": Tensor[N, 4)
                "labels": Tensor[N]
        """
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_cls, device=boxes_h.device)

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(boxes_h, targets["boxes_h"]),
            box_ops.box_iou(boxes_o, targets["boxes_o"])
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets["labels"][y]] = 1

        return labels

    def compute_prior_scores(self, x, y, scores, object_class):
        """
        Arguments:
            x(Tensor[M]): Indices of human boxes (paired)
            y(Tensor[M]): Indices of object boxes (paired)
            scores(Tensor[N])
            object_class(Tensor[N])
        """
        prior = torch.zeros(len(x), self.num_cls, device=scores.device)

        # Product of human and object detection scores with LIS
        prod = LIS(scores[x]) * LIS(scores[y])

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior[pair_idx, flat_target_idx] = prod[pair_idx]

        return prior

    def forward(self, features, box_features, box_coords, box_labels, box_scores, targets=None):
        """
        Arguments:
            features(OrderedDict[Tensor]): Image pyramid with different levels
            box_features(Tensor[M, R])
            box_coords(List[Tensor])
            box_labels(List[Tensor])
            box_scores(List[Tensor])
            targets(list[dict]): Interaction targets with the following keys
                "boxes_h": Tensor[N, 4]
                "boxes_o": Tensor[N, 4]
                "labels": Tensor[N]
        Returns:
            all_box_pair_features(Tensor[P, 2*R])
            all_boxes_h(list[Tensor])
            all_boxes_o(list[Tensor])
            all_labels(list[Tensor])
            all_prior(Tensor[P, K])
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"

        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]
        
        counter = 0
        all_boxes_h = []; all_boxes_o = []
        all_labels = []; all_prior = []
        all_box_pair_features = []
        for b_idx, (coords, labels, scores) in enumerate(zip(box_coords, box_labels, box_scores)):
            n = num_boxes[b_idx]
            device = box_features.device

            human_box_idx = torch.nonzero(labels == self.human_idx).squeeze(1).tolist()
            # Skip image when there are no detected human or object instances
            if n == 0 or len(human_box_idx) == 0:
                continue
            # Permute the boxes so that humans are on the top
            permutation = torch.cat([
                torch.as_tensor(human_box_idx, device=device),
                torch.as_tensor([i for i in range(n) if i not in human_box_idx], device=device)
            ])
            coords = coords[permutation]
            labels = labels[permutation]
            scores = scores[permutation]
            node_encodings = box_features[counter: counter+n][permutation]

            n_h = len(human_box_idx)
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(torch.arange(n_h), torch.arange(n), device=device)
            x, y = torch.nonzero(x != y).unbind(1)

            for i in range(self.num_iter):
                # Compute weights of each edge
                weights = self.adjacency(torch.cat([
                    node_encodings[x],
                    node_encodings[y]
                ], 1))
                # Construct adjacency matrix
                # Diagonal entries are set to zero
                adjacency_matrix = torch.zeros([n_h, n], device=device)
                adjacency_matrix[x, y] = torch.sigmoid(weights)

                # Update human nodes
                node_encodings[:n_h] = self.sub_update(torch.cat([
                    node_encodings[:n_h],
                    torch.mm(adjacency_matrix, self.obj_to_sub(node_encodings))
                ], 1))

                # Update object nodes (including human nodes)
                node_encodings = self.obj_update(torch.cat([
                    node_encodings,
                    torch.mm(adjacency_matrix.t(), self.sub_to_obj(node_encodings[:n_h]))
                ], 1))

            if self.training:
                all_labels.append(self.associate_with_ground_truth(
                    coords[x], coords[y], targets[b_idx])
                )
                
            all_box_pair_features.append(torch.cat([
                node_encodings[x], node_encodings[y]
            ], 1))
            all_boxes_h.append(coords[x])
            all_boxes_o.append(coords[y])
            # The prior score is the product between edge weights and the
            # pre-computed object detection scores with LIS
            all_prior.append(
                adjacency_matrix[x, y, None] *
                self.compute_prior_scores(x, y, scores, labels)
            )

            counter += n

        all_box_pair_features = torch.cat(all_box_pair_features)
        all_prior = torch.cat(all_prior)

        return all_box_pair_features, all_boxes_h, all_boxes_o, all_labels, all_prior

class BoxPairPredictor(nn.Module):
    def __init__(self, input_size, representation_size, num_classes):
        self.predictor = nn.Sequential(
            nn.Linear(input_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, num_classes)
        )
    def forward(self, x, prior):
        return torch.sigmoid(self.predictor(x)) * prior

class InteractGraphNet(models.GenericHOINetwork):
    def __init__(self,
            object_to_action, human_idx,
            # Backbone parameters
            backbone_name="resnet50", pretrained=True,
            # Pooler parameters
            output_size=7, sampling_ratio=2,
            # Box pair head parameters
            node_encoding_size=1024,
            representation_size=1024,
            num_classes=117,
            fg_iou_thresh=0.5,
            num_iterations=1,
            # Transformation parameters
            min_size=800, max_size=1333,
            image_mean=None, image_std=None,
            # Preprocessing parameters
            box_score_thresh_h=0.6,
            box_score_thresh_o=0.3,
            box_nms_thresh=0.5
            ):

        backbone = models.fasterrcnn_resnet_fpn(backbone_name,
            pretrained=pretrained).backbone

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=output_size,
            sampling_ratio=sampling_ratio
        )

        box_pair_head = InteractGraph(
            node_encoding_size=node_encoding_size,
            representation_size=representation_size,
            num_cls=num_classes,
            human_idx=human_idx,
            object_class_to_target_class=object_to_action,
            fg_iou_thresh=fg_iou_thresh,
            num_iter=num_iterations
        )

        box_pair_predictor = BoxPairPredictor(
            input_size=node_encoding_size * 2,
            representation_size=representation_size,
            num_classes=num_classes
        )

        interaction_head = models.InteractionHead(
            box_roi_pool=box_roi_pool,
            box_pair_head=box_pair_head,
            box_pair_predictor=box_pair_predictor,
            human_idx=human_idx,
            box_score_thresh_h=box_score_thresh_h,
            box_score_thresh_o=box_score_thresh_o,
            box_nms_thresh=box_nms_thresh
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = models.HOINetworkTransform(min_size, max_size,
            image_mean, image_std)

        super().__init__(backbone, interaction_head, transform)

    def state_dict(self):
        """Override method to only return state dict of the interaction head"""
        return self.interaction_head.state_dict()
    def load_state_dict(self, x):
        """Override method to only load state dict of the interaction head"""
        self.interaction_head.load_state_dict(x)
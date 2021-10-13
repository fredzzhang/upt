"""
Interaction head and its submodules

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops

from torch.nn import Module
from torch import nn, Tensor
from pocket.ops import Flatten
from typing import Optional, List, Tuple
from collections import OrderedDict

import pocket

from ops import compute_spatial_encodings, binary_focal_loss

class MultiBranchFusion(Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        representation_size: int, cardinality: int
    ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(representation_size / cardinality)
        assert sub_repr_size * cardinality == representation_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, representation_size)
            for _ in range(cardinality)
        ])
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))

class UnaryLayer(Module):
    def __init__(self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        return_weights: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The given hidden size {hidden_size} should be divisible by "
                f"the number of attention heads {num_heads}."
            )
        self.sub_hidden_size = int(hidden_size / num_heads)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.return_weights = return_weights

        self.unary = nn.Linear(hidden_size, hidden_size)
        self.pairwise = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.ModuleList([nn.Linear(3 * self.sub_hidden_size, 1) for _ in range(num_heads)])
        self.message = nn.ModuleList([nn.Linear(self.sub_hidden_size, self.sub_hidden_size) for _ in range(num_heads)])
        self.aggregate = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def reshape(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.sub_hidden_size
        )
        x = x.view(*new_x_shape)
        if len(new_x_shape) == 3:
            return x.permute(1, 0, 2)
        elif len(new_x_shape) == 4:
            return x.permute(2, 0, 1, 3)
        else:
            raise ValueError("Incorrect tensor shape")

    def forward(self, x: Tensor, y: Tensor):
        device = x.device
        n = len(x)

        u = F.relu(self.unary(x))
        p = F.relu(self.pairwise(y))

        # Unary features (H, N, L)
        u_r = self.reshape(u)
        # Pairwise features (H, N, N, L)
        p_r = self.reshape(p)

        i, j = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )

        # Features used to compute attention (H, N, N, 3L)
        attn_features = torch.cat([
            u_r[:, i], u_r[:, j], p_r
        ], dim=-1)
        # Attention weights (H,) (N, N, 1)
        weights = [
            F.softmax(l(f), dim=0) for f, l
            in zip(attn_features, self.attn)
        ]
        # Repeated unary feaures along the third dimension (H, N, N, L)
        u_r_repeat = u_r.unsqueeze(dim=2).repeat(1, 1, n, 1)
        messages = [
            l(f_1 * f_2) for f_1, f_2, l
            in zip(u_r_repeat, p_r, self.message)
        ]

        aggreagted_messages = self.aggregate(F.relu(
            torch.cat([
                (w * m).sum(dim=0) for w, m
                in zip(weights, messages)
            ], dim=-1)
        ))
        x = self.norm(x + aggreagted_messages)

        if self.return_weights:
            attn = weights
        else:
            attn = None

        return x, attn
class PairwiseLayer(Module):
    def __init__(self,
        hidden_size: int = 1024,
        num_heads: int = 8
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The given hidden size {hidden_size} should be divisible by "
                f"the number of attention heads {num_heads}."
            )
        self.sub_hidden_size = int(hidden_size / num_heads)
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.unary = nn.Linear(hidden_size, hidden_size)
        self.pairwise = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.ModuleList([nn.Linear(3 * self.sub_hidden_size, 1) for _ in range(num_heads)])

    def reshape(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.sub_hidden_size
        )
        x = x.view(*new_x_shape)
        if len(new_x_shape) == 3:
            return x.permute(1, 0, 2)
        elif len(new_x_shape) == 4:
            return x.permute(2, 0, 1, 3)
        else:
            raise ValueError("Incorrect tensor shape")

    def forward(self, x: Tensor, y: Tensor, i: Tensor, j: Tensor):

        u = F.relu(self.unary(x))
        p = F.relu(self.pairwise(y))

        # Unary features (H, N, L)
        u_r = self.reshape(u)
        # Pairwise features (H, N, N, L)
        p_r = self.reshape(p)

        # Features used to compute attention (H, M, 3L)
        attn_features = torch.cat([
            u_r[:, i], u_r[:, j], p_r[:, i, j]
        ], dim=-1)
        # Attention weights (H, M)
        weights = torch.stack([
            l(f) for f, l
            in zip(attn_features, self.attn)
        ]).squeeze(-1)

        return weights

class InteractionHead(Module):
    """Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_roi_pool: Module
        Module that performs RoI pooling or its variants
    box_pair_head: Module
        Module that constructs and computes box pair features
    box_pair_suppressor: Module
        Module that computes unary weights for each box pair
    box_pair_predictor: Module
        Module that classifies box pairs
    human_idx: int
        The index of human/person class in all objects
    num_classes: int
        Number of target classes
    box_nms_thresh: float, default: 0.5
        Threshold used for non-maximum suppression
    box_score_thresh: float, default: 0.2
        Threshold used to filter out low-quality boxes
    max_human: int, default: 15
        Number of human detections to keep in each image
    max_object: int, default: 15
        Number of object (excluding human) detections to keep in each image
    distributed: bool, default: False
        Whether the model is trained under distributed data parallel. If True,
        the number of positive logits will be averaged across all subprocesses
    """
    def __init__(self,
        # Network components
        box_roi_pool: Module,
        box_pair_predictor: Module,
        out_channels: int,
        roi_pool_size: int,
        node_encoding_size: int, 
        representation_size: int, 
        # Dataset properties
        human_idx: int,
        num_classes: int,
        object_class_to_target_class: List[list],
        # Hyperparameters
        num_iter: int = 2,
        box_nms_thresh: float = 0.5,
        box_score_thresh: float = 0.2,
        fg_iou_thresh: float = 0.5,
        max_human: int = 15,
        max_object: int = 15,
        # Misc
        distributed: bool = False,
    ) -> None:
        super().__init__()

        self.box_roi_pool = box_roi_pool
        self.box_pair_predictor = box_pair_predictor

        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size

        self.human_idx = human_idx
        self.num_classes = num_classes
        self.object_class_to_target_class = object_class_to_target_class

        self.num_iter = num_iter

        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.max_human = max_human
        self.max_object = max_object

        self.distributed = distributed

        # Box head to map RoI features to low dimensional
        self.box_head = nn.Sequential(
            Flatten(start_dim=1),
            nn.Linear(out_channels * roi_pool_size ** 2, node_encoding_size),
            nn.ReLU(),
            nn.Linear(node_encoding_size, node_encoding_size),
            nn.ReLU()
        )

        # Map spatial encodings to the same dimension as appearance features
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        self.matching_layer = UnaryLayer(
            hidden_size=representation_size,
            return_weights=True
        )
        self.attention_layer = PairwiseLayer(
            hidden_size=representation_size
        )
        self.feature_head = pocket.models.TransformerEncoderLayer(
            hidden_size=representation_size * 2,
            return_weights=True
        )

        # Spatial attention head
        self.attention_head = MultiBranchFusion(
            node_encoding_size * 2,
            1024, representation_size,
            cardinality=16
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Attention head for global features
        self.attention_head_g = MultiBranchFusion(
            256, 1024,
            representation_size, cardinality=16
        )

    def preprocess(self,
        detections: List[dict],
        targets: List[dict],
        append_gt: Optional[bool] = None
    ) -> None:

        results = []
        for b_idx, detection in enumerate(detections):
            boxes = detection['boxes']
            labels = detection['labels']
            scores = detection['scores']

            # Append ground truth during training
            if append_gt is None:
                append_gt = self.training
            if append_gt:
                target = targets[b_idx]
                n = target["boxes_h"].shape[0]
                boxes = torch.cat([target["boxes_h"], target["boxes_o"], boxes])
                scores = torch.cat([torch.ones(2 * n, device=scores.device), scores])
                labels = torch.cat([
                    self.human_idx * torch.ones(n, device=labels.device).long(),
                    target["object"],
                    labels
                ])

            # Remove low scoring examples
            active_idx = torch.nonzero(
                scores >= self.box_score_thresh
            ).squeeze(1)
            # Class-wise non-maximum suppression
            keep_idx = box_ops.batched_nms(
                boxes[active_idx],
                scores[active_idx],
                labels[active_idx],
                self.box_nms_thresh
            )
            active_idx = active_idx[keep_idx]
            # Sort detections by scores
            sorted_idx = torch.argsort(scores[active_idx], descending=True)
            active_idx = active_idx[sorted_idx]
            # Keep a fixed number of detections
            h_idx = torch.nonzero(labels[active_idx] == self.human_idx).squeeze(1)
            o_idx = torch.nonzero(labels[active_idx] != self.human_idx).squeeze(1)
            if len(h_idx) > self.max_human:
                h_idx = h_idx[:self.max_human]
            if len(o_idx) > self.max_object:
                o_idx = o_idx[:self.max_object]
            # Permute humans to the top
            keep_idx = torch.cat([h_idx, o_idx])
            active_idx = active_idx[keep_idx]

            results.append(dict(
                boxes=boxes[active_idx].view(-1, 4),
                labels=labels[active_idx].view(-1),
                scores=scores[active_idx].view(-1)
            
            ))

        return results

    def associate_with_ground_truth(self,
        boxes_h: Tensor,
        boxes_o: Tensor,
        targets: List[dict]
    ) -> Tensor:
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(boxes_h, targets["boxes_h"]),
            box_ops.box_iou(boxes_o, targets["boxes_o"])
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets["labels"][y]] = 1

        return labels

    def compute_prior_scores(self,
        x: Tensor, y: Tensor,
        scores: Tensor,
        object_class: Tensor
    ) -> Tensor:
        """
        Parameters:
        -----------
            x: Tensor[M]
                Indices of human boxes (paired)
            y: Tensor[M]
                Indices of object boxes (paired)
            scores: Tensor[N]
                Object detection scores (before pairing)
            object_class: Tensor[N]
                Object class indices (before pairing)
        """
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def compute_interaction_classification_loss(self, results: List[dict]) -> Tensor:
        scores = []; labels = []
        for result in results:
            scores.append(result['scores'])
            labels.append(result['labels'])

        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            torch.cat(scores), labels, reduction='sum', gamma=0.2
        )
        return loss / n_p

    def compute_interactiveness_loss(self, results: List[dict]) -> Tensor:
        weights = []; labels = []
        for result in results:
            weights.append(result['weights'])
            labels.append(result['unary_labels'])

        weights = torch.cat(weights)
        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            weights, labels, reduction='sum', gamma=2.0
        )
        return loss / n_p

    def postprocess(self,
        logits_p: Tensor,
        logits_s: Tensor,
        prior: List[Tensor],
        box_coords: List[Tensor],
        boxes_h: List[Tensor],
        boxes_o: List[Tensor],
        object_class: List[Tensor],
        labels: List[Tensor],
        attn_maps: List[list]
    ) -> List[dict]:
        """
        Parameters:
        -----------
        logits_p: Tensor
            (N, K) Classification logits on each action for all box pairs
        logits_s: Tensor
            (N, 1) Logits for unary weights
        prior: List[Tensor]
            Prior scores organised by images. Each tensor has shape (2, M, K).
            M could be different for different images
        boxes_h: List[Tensor]
            Human bounding box coordinates organised by images (M, 4)
        boxes_o: List[Tensor]
            Object bounding box coordinates organised by images (M, 4)
        object_classes: List[Tensor]
            Object indices for each pair organised by images (M,)
        labels: List[Tensor]
            Binary labels on each action organised by images (M, K)

        Returns:
        --------
        results: List[dict]
            Results organised by images, with keys as below
            `boxes_h`: Tensor[M, 4]
            `boxes_o`: Tensor[M, 4]
            `index`: Tensor[L]
                Expanded indices of box pairs for each predicted action
            `prediction`: Tensor[L]
                Expanded indices of predicted actions
            `scores`: Tensor[L]
                Scores for each predicted action
            `object`: Tensor[M]
                Object indices for each pair
            `prior`: Tensor[2, L]
                Prior scores for expanded pairs
            `weights`: Tensor[M]
                Unary weights for each box pair
            `labels`: Tensor[L], optional
                Binary labels on each action
            `unary_labels`: Tensor[M], optional
                Labels for the unary weights
        """
        num_boxes = [len(b) for b in boxes_h]

        weights = torch.sigmoid(logits_s).mean(0)
        scores = torch.sigmoid(logits_p)
        weights = weights.split(num_boxes)
        scores = scores.split(num_boxes)
        if len(labels) == 0:
            labels = [None for _ in range(len(num_boxes))]

        results = []
        for w, s, p, b, b_h, b_o, o, l, a in zip(
            weights, scores, prior, box_coords, boxes_h, boxes_o, object_class, labels, attn_maps
        ):
            # Keep valid classes
            x, y = torch.nonzero(p[0]).unbind(1)

            result_dict = dict(
                boxes=b, boxes_h=b_h[x], boxes_o=b_o[x], prediction=y,
                pairs=[(i.item(), j.item()) for i, j in zip(b_h, b_o)],
                scores=s[x, y] * p[:, x, y].prod(dim=0) * w[x].detach(),
                object=o[x], prior=p[:, x, y], weights=w, attn_maps=a
            )
            # If binary labels are provided
            if l is not None:
                result_dict['labels'] = l[x, y]
                result_dict['unary_labels'] = l.sum(dim=1).clamp(max=1)

            results.append(result_dict)

        return results

    def forward(self,
        features: OrderedDict,
        detections: List[dict],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        detections: List[dict]
            Object detections with the following keys
            `boxes`: Tensor[N, 4]
            `labels`: Tensor[N]
            `scores`: Tensor[N]
        image_shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        targets: List[dict], optional
            Interaction targets with the following keys
            `boxes_h`: Tensor[G, 4]
            `boxes_o`: Tensor[G, 4]
            `object`: Tensor[G]
                Object class indices for each pair
            `labels`: Tensor[G]
                Target class indices for each pair

        Returns:
        --------
        results: List[dict]
            Results organised by images. During training the loss dict is appended to the
            end of the list, resulting in the length being larger than the number of images
            by one. For the result dict of each image, refer to `postprocess` for documentation.
            The loss dict has two keys
            `hoi_loss`: Tensor
                Loss for HOI classification
            `interactiveness_loss`: Tensor
                Loss incurred on learned unary weights
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        detections = self.preprocess(detections, targets)

        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        box_scores = [detection['scores'] for detection in detections]

        box_features = self.box_roi_pool(features, box_coords, image_shapes)
        global_features = self.avg_pool(features['3']).flatten(start_dim=1)
        box_features = self.box_head(box_features)

        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]

        counter = 0
        boxes_h_collated = []; boxes_o_collated = []
        object_class_collated = []; labels_collated = []
        prior_collated = []; pairwise_features_collated = []
        attn_maps_collated = []; unary_logits_collated = []

        for b_idx, (coords, labels, scores) in enumerate(zip(box_coords, box_labels, box_scores)):
            n = num_boxes[b_idx]
            device = box_features.device

            n_h = torch.sum(labels == self.human_idx).item()
            # Skip image when there are no detected human or object instances
            # and when there is only one detected instance
            if n_h == 0 or n <= 1:
                pairwise_features_collated.append(torch.zeros(
                    0, 2 * self.representation_size,
                    device=device)
                )
                boxes_h_collated.append(torch.zeros(0, 4, device=device))
                boxes_o_collated.append(torch.zeros(0, 4, device=device))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                labels_collated.append(torch.zeros(0, self.num_classes, device=device))
                unary_logits_collated.append(torch.zeros(self.attention_layer.num_heads, 0, device=device))
                continue
            if not torch.all(labels[:n_h]==self.human_idx):
                raise ValueError("Human detections are not permuted to the top")

            node_encodings = box_features[counter: counter+n]
            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            # Human nodes have been duplicated and will be treated independently
            # of the humans included amongst object nodes
            x = x.flatten(); y = y.flatten()

            # Compute spatial features
            box_pair_spatial = compute_spatial_encodings(
                [coords[x]], [coords[y]], [image_shapes[b_idx]]
            )
            box_pair_spatial = self.spatial_head(box_pair_spatial)
            # Reshape the spatial features
            box_pair_spatial_reshaped = box_pair_spatial.reshape(n, n, -1)

            # Run the matching layer
            node_encodings, attn_data = self.matching_layer(node_encodings, box_pair_spatial_reshaped)
            # Run the pairing layer
            pairing_weights = self.attention_layer(node_encodings, box_pair_spatial_reshaped, x_keep, y_keep)

            box_pair_features = torch.cat([
                self.attention_head(
                    torch.cat([
                        node_encodings[x_keep],
                        node_encodings[y_keep]
                        ], 1),
                    box_pair_spatial_reshaped[x_keep, y_keep]
                ), self.attention_head_g(
                    global_features[b_idx, None],
                    box_pair_spatial_reshaped[x_keep, y_keep])
            ], dim=1)
            # Run the feature head
            box_pair_features, attn_data_2 = self.feature_head(box_pair_features)

            if targets is not None:
                labels_collated.append(self.associate_with_ground_truth(
                    coords[x_keep], coords[y_keep], targets[b_idx])
                )
            pairwise_features_collated.append(box_pair_features)
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            # The prior score is the product of the object detection scores
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            attn_maps_collated.append((attn_data, attn_data_2))
            unary_logits_collated.append(pairing_weights)

            counter += n

        pairwise_features_collated = torch.cat(pairwise_features_collated)
        logits = self.box_pair_predictor(pairwise_features_collated)
        unary_logits_collated = torch.cat(unary_logits_collated, dim=1)

        results = self.postprocess(
            logits, unary_logits_collated, prior_collated,
            box_coords, boxes_h_collated, boxes_o_collated,
            object_class_collated, labels_collated, attn_maps_collated
        )

        if self.training:
            loss_dict = dict(
                hoi_loss=self.compute_interaction_classification_loss(results),
                interactiveness_loss=self.compute_interactiveness_loss(results)
            )
            results.append(loss_dict)

        return results

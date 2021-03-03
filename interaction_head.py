"""
Interaction head and its variants

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops

from torch import nn
from pocket.ops import Flatten

from ops import LIS, compute_spatial_encodings, binary_focal_loss

class InteractionHead(nn.Module):
    """Interaction head that constructs and classifies box pairs

    Arguments:

    [REQUIRES ARGS]
        box_roi_pool(nn.Module): Module that performs RoI pooling or its variants
        box_pair_head(nn.Module): Module that constructs and computes box pair features
        box_pair_predictor(nn.Module): Module that classifies box pairs
        human_idx(int): The index of human/person class in all objects
        num_classes(int): Number of target classes
        box_nms_thresh(float): Threshold used for non-maximum suppression
        max_human(int): Number of human detections to keep in each image
        max_object(int): Number of object (excluding human) detections to keep in each image

    [OPTIONAL ARGS]
        box_nms_thresh(float): NMS threshold
    """
    def __init__(self,
                box_roi_pool,
                box_head,
                cls_head,
                box_pair_head,
                box_pair_predictor,
                human_idx,
                num_classes,
                box_nms_thresh=0.5,
                box_score_thresh=0.05,
                max_human=10,
                max_object=10,
                distributed=False
                ):
        
        super().__init__()

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.cls_head = cls_head
        self.box_pair_head = box_pair_head
        self.box_pair_predictor = box_pair_predictor

        self.num_classes = num_classes
        self.human_idx = human_idx
        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh

        self.max_human = max_human
        self.max_object = max_object
        self.distributed = distributed

        with open('./hicodet/coco80tohico80.json', 'r') as f:
            conversion = json.load(f)
        # Denote background class with index 80, which should throw
        # an index error if the prediction was somehow not removed
        conversion = torch.as_tensor(
            [80,] + list(conversion.values())
        )
        # Reverse the conversion
        idx = torch.argsort(conversion)
        self.conversion = torch.arange(81)[idx].tolist()

    def append_ground_truth(self, box_coords, targets):
        """
        Parameters:
            box_coords: List[Tensor]
            targets: List[dict]
        """
        augmented_boxes = []
        for c, t in zip(box_coords, targets):
            boxes = torch.cat([
                t['boxes_h'], t['boxes_o'], c
            ], dim=0)
            augmented_boxes.append(boxes)

        return augmented_boxes

    def preprocess(self, box_coords, box_scores, num_gt_boxes):
        """
        box_coords: List[Tensor]
        box_scores: Tensor
        """
        return_coords = []
        return_scores = []
        return_labels = []
        return_idx = []
        # Remove scores predicted for the background class
        box_scores = box_scores[:, :-1]
        counter = 0
        for boxes, n_gt in zip(box_coords, num_gt_boxes):
            n = boxes.shape[0]
            scores, labels = torch.max(
                box_scores[counter: counter + n, :], dim=1
            )
            counter += n
            # Clamp the scores of the ground truth boxes to keep them in
            scores[:n_gt].clamp_(min=self.box_score_thresh)
            # Remove background predictions and low scoring examples
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

            return_coords.append(boxes[active_idx].view(-1, 4))
            return_scores.append(scores[active_idx].view(-1))
            return_labels.append(labels[active_idx].view(-1))
            return_idx.append(active_idx)

        return return_coords, return_scores, return_labels, return_idx

    def compute_object_classification_loss(self, boxes, logits, targets):
        labels = []
        for b, t in zip(boxes, targets):
            if len(b) == 0:
                continue
            gt_boxes = torch.cat([
                t['boxes_h'],
                t['boxes_o']
            ])
            gt_classes = torch.cat([
                torch.ones_like(t['object']) * self.human_idx,
                t['object']
            ])
            # Set the default class to background
            l = 80 * torch.ones(len(b), dtype=torch.int64, device=b.device)
            
            iou = box_ops.box_iou(b, gt_boxes)
            max_iou, gt_idx = torch.max(iou, dim=1)
            match = torch.nonzero(
                max_iou >= self.box_pair_head.fg_iou_thresh
            ).squeeze()
            l[match] = gt_classes[gt_idx[match]]

            labels.append(l)

        logits = torch.cat(logits)
        labels = torch.cat(labels)

        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_interaction_classification_loss(self, results):
        """
        Arguments:
            results(List[dict]): See output of self.postprocess
        """
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
            torch.cat(scores), labels, reduction='sum', gamma=0.5
        )
        return loss / n_p

    def postprocess(self, logits, prior, boxes_h, boxes_o, object_class, labels):
        """
        Arguments:
            logits(Tensor[N,K]): Pre-sigmoid logits for target classes
            prior(List[Tensor[M,K]]): Prior scores organised on a per-image basis
            boxes_h(List[Tensor[M,4]])
            boxes_o(List[Tensor[M,4]])
            object_class(List[Tensor[M]])
            labels(List[Tensor[M,K]])
        Returns:
            List[dict] with the following keys
                'boxes_h': Tensor[M,4]
                'boxes_o': Tensor[M,4]
                'index': Tensor[L]: Indices of boxes for each prediction
                'prediction': Tensor[L]: Predicted target class indices
                'scores': Tensor[L]: Predicted scores
                'object': Tensor[M]: Object class indices
                'labels': Tensor[L]: Binary labels for each prediction

        """
        num_boxes = [len(p) for p in prior]
        scores = torch.sigmoid(logits)
        scores = scores.split(num_boxes)
        if len(labels) == 0:
            labels = [[] for _ in range(len(num_boxes))]

        results = []
        for s, p, b_h, b_o, o, l in zip(
            scores, prior, boxes_h, boxes_o, object_class, labels
        ):
            # Keep valid classes
            x, y = torch.nonzero(p).unbind(1)

            result_dict = dict(
                boxes_h=b_h, boxes_o=b_o,
                index=x, prediction=y,
                scores=s[x, y] * p[x, y],
                object=o, prior=p[x, y]
            )
            # If binary labels are provided
            if len(l):
                result_dict["labels"] = l[x, y]

            results.append(result_dict)

        return results

    def forward(self, features, detections, image_shapes, targets=None):
        """
        Arguments:
            features(OrderedDict[Tensor]): Image pyramid with different levels
            detections(list[dict]): Object detections with following keys 
                "boxes": Tensor[N, 4]
                "labels": Tensor[N]
                "scores": Tensor[N]
            image_shapes(List[Tuple[height, width]])
            targets(list[dict]): Interaction targets with the following keys
                "boxes_h": Tensor[N, 4]
                "boxes_o": Tensor[N, 4]
                "object": Tensor[N] Object class index for the object in each pair
                "labels": Tensor[N] Target class index for each pair
        Returns:
            results(list[dict]): During evaluation, return dicts of detected interacitons
                "boxes_h": Tensor[M, 4]
                "boxes_o": Tensor[M, 4]
                "object": Tensor[M] Object types in each pair
                "labels": list(Tensor) The predicted label indices. A list of length M.
                "scores": list(Tensor) The predcited scores. A list of length M. 
                "gt_labels": list(Tensor): Binary labels. One if predicted label index is correct,
                    zero otherwise. This is only returned when targets are given
            During training, the classification loss is appended to the end of the list
        """
        box_coords = [detection['boxes'] for detection in detections]
        if self.training:
            assert targets is not None, "Targets should be passed during training."
            box_coords = self.append_ground_truth(box_coords, targets)
            num_gt_boxes = [len(t['boxes_h']) + len(t['boxes_o']) for t in targets]
        else:
            num_gt_boxes = [0 for _ in range(len(box_coords))]

        box_features = self.box_roi_pool(features, box_coords, image_shapes)
        box_features = box_features.flatten(start_dim=1)
        box_features = self.box_head(box_features)
        box_logits = self.cls_head(box_features)
        # Re-arrange object classes in HICO standard
        box_logits = box_logits[:, self.conversion]
        box_scores = F.softmax(box_logits, dim=1).detach()

        n_boxes = [c.shape[0] for c in box_coords]
        box_logits = box_logits.split(n_boxes)
        box_features = box_features.split(n_boxes)

        box_coords, box_scores, box_labels, active_idx = self.preprocess(
            box_coords, box_scores, num_gt_boxes
        )
        # Update box features and logits
        box_logits = [l[idx] for l, idx in zip(box_logits, active_idx)]
        box_features = [f[idx] for f, idx in zip(box_features, active_idx)]

        box_pair_features, boxes_h, boxes_o, object_class,\
        box_pair_labels, box_pair_prior = self.box_pair_head(
            features, image_shapes, box_features,
            box_coords, box_labels, box_scores, targets
        )

        # No valid human-object pairs were formed
        if len(box_pair_features) == 0:
            return None
        else:
            box_pair_features = torch.cat(box_pair_features)

        logits = self.box_pair_predictor(box_pair_features)

        results = self.postprocess(
            logits, box_pair_prior, boxes_h, boxes_o,
            object_class, box_pair_labels
        )

        if self.training:
            loss_dict = dict(
                hoi_loss=self.compute_interaction_classification_loss(results),
                object_loss=self.compute_object_classification_loss(
                    box_coords, box_logits, targets)
            )
            results.append(loss_dict)

        return results

class AttentionHead(nn.Module):
    def __init__(self, appearance_size, spatial_size, representation_size, cardinality):
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
    def forward(self, appearance, spatial):
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))

class MessageAttentionHead(AttentionHead):
    def __init__(self, appearance_size, spatial_size, representation_size, node_type, cardinality):
        super().__init__(appearance_size, spatial_size, representation_size, cardinality)

        if node_type == 'human':
            self._forward_method = self._forward_human_nodes
        elif node_type == 'object':
            self._forward_method = self._forward_object_nodes
        else:
            raise ValueError("Unknown node type \"{}\"".format(node_type))

    def _forward_human_nodes(self, appearance, spatial):
        n_h, n = spatial.shape[:2]
        assert len(appearance) == n_h, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n, 1, 1)
                * fc_2(spatial).permute([1, 0, 2])
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)
    def _forward_object_nodes(self, appearance, spatial):
        n_h, n = spatial.shape[:2]
        assert len(appearance) == n, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n_h, 1, 1)
                * fc_2(spatial)
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)

    def forward(self, *args):
        return self._forward_method(*args)

class GraphHead(nn.Module):
    def __init__(self,
                out_channels,
                roi_pool_size,
                node_encoding_size, 
                representation_size, 
                num_cls, human_idx,
                object_class_to_target_class,
                fg_iou_thresh=0.5,
                num_iter=1):

        super().__init__()

        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size

        self.num_cls = num_cls
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter

        # Compute adjacency matrix
        self.adjacency = nn.Linear(representation_size, 1)

        # Compute messages
        self.sub_to_obj = MessageAttentionHead(
            node_encoding_size, 1024,
            representation_size, node_type='human',
            cardinality=16
        )
        self.obj_to_sub = MessageAttentionHead(
            node_encoding_size, 1024,
            representation_size, node_type='object',
            cardinality=16
        )

        self.norm_h = nn.LayerNorm(node_encoding_size)
        self.norm_o = nn.LayerNorm(node_encoding_size)

        # Map spatial encodings to the same dimension as appearance features
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        # Spatial attention head
        self.attention_head = AttentionHead(
            node_encoding_size * 2,
            1024, representation_size,
            cardinality=16
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Attention head for global features
        self.attention_head_g = AttentionHead(
            256, 1024,
            representation_size, cardinality=16
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
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior[pair_idx, flat_target_idx] = prod[pair_idx]

        return prior

    def forward(self,
        features, image_shapes, box_features, box_coords,
        box_labels, box_scores, targets=None
    ):
        """
        Arguments:
            features(OrderedDict[Tensor]): Image pyramid with different levels
            box_features(List[Tensor])
            image_shapes(List[Tuple[height, width]])
            box_coords(List[Tensor])
            box_labels(List[Tensor])
            box_scores(List[Tensor])
            targets(list[dict]): Interaction targets with the following keys
                "boxes_h": Tensor[N, 4]
                "boxes_o": Tensor[N, 4]
                "labels": Tensor[N]
        Returns:
            all_box_pair_features(list[Tensor])
            all_boxes_h(list[Tensor])
            all_boxes_o(list[Tensor])
            all_object_class(list[Tensor])
            all_labels(list[Tensor])
            all_prior(list[Tensor])
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"

        global_features = self.avg_pool(features['3']).flatten(start_dim=1)
        
        counter = 0
        all_boxes_h = []; all_boxes_o = []; all_object_class = []
        all_labels = []; all_prior = []
        all_box_pair_features = []
        for b_idx, (coords, labels, scores, features) in enumerate(
            zip(box_coords, box_labels, box_scores, box_features)
        ):
            n = coords.shape[0]
            device = features.device

            n_h = torch.sum(labels == self.human_idx).item()
            # Skip image when there are no detected human or object instances
            # and when there is only one detected instance
            if n_h == 0 or n <= 1:
                continue
            if not torch.all(labels[:n_h]==self.human_idx):
                raise ValueError("Human detections are not permuted to the top")

            node_encodings = features
            # Duplicate human nodes
            h_node_encodings = node_encodings[:n_h]
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(
                torch.arange(n_h, device=device),
                torch.arange(n, device=device)
            )
            # Remove pairs consisting of the same human instance
            x_keep, y_keep = torch.nonzero(x != y).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                continue
            # Human nodes have been duplicated and will be treated independently
            # of the humans included amongst object nodes
            x = x.flatten(); y = y.flatten()

            # Compute spatial features
            box_pair_spatial = compute_spatial_encodings(
                [coords[x]], [coords[y]], [image_shapes[b_idx]]
            )
            box_pair_spatial = self.spatial_head(box_pair_spatial)
            # Reshape the spatial features
            box_pair_spatial_reshaped = box_pair_spatial.reshape(n_h, n, -1)

            adjacency_matrix = torch.ones(n_h, n, device=device)
            for _ in range(self.num_iter):
                # Compute weights of each edge
                weights = self.attention_head(
                    torch.cat([
                        h_node_encodings[x],
                        node_encodings[y]
                    ], 1),
                    box_pair_spatial
                )
                adjacency_matrix = self.adjacency(weights).reshape(n_h, n)

                # Update human nodes
                messages_to_h = F.relu(torch.sum(
                    adjacency_matrix.softmax(dim=1)[..., None] *
                    self.obj_to_sub(
                        node_encodings,
                        box_pair_spatial_reshaped
                    ), dim=1)
                )
                h_node_encodings = self.norm_h(
                    h_node_encodings + messages_to_h
                )

                # Update object nodes (including human nodes)
                messages_to_o = F.relu(torch.sum(
                    adjacency_matrix.t().softmax(dim=1)[..., None] *
                    self.sub_to_obj(
                        h_node_encodings,
                        box_pair_spatial_reshaped
                    ), dim=1)
                )
                node_encodings = self.norm_o(
                    node_encodings + messages_to_o
                )

            if targets is not None:
                all_labels.append(self.associate_with_ground_truth(
                    coords[x_keep], coords[y_keep], targets[b_idx])
                )
                
            all_box_pair_features.append(torch.cat([
                self.attention_head(
                    torch.cat([
                        h_node_encodings[x_keep],
                        node_encodings[y_keep]
                        ], 1),
                    box_pair_spatial_reshaped[x_keep, y_keep]
                ), self.attention_head_g(
                    global_features[b_idx, None],
                    box_pair_spatial_reshaped[x_keep, y_keep])
            ], dim=1))
            all_boxes_h.append(coords[x_keep])
            all_boxes_o.append(coords[y_keep])
            all_object_class.append(labels[y_keep])
            # The prior score is the product of the pre-computed object detection scores with LIS
            all_prior.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )

            counter += n

        return all_box_pair_features, all_boxes_h, all_boxes_o, \
            all_object_class, all_labels, all_prior

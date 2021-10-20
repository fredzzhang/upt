"""
Human-object interaction detector

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.distributed as dist

import sys
from torch.nn.functional import binary_cross_entropy
from torchvision.models import detection
sys.path.append('detr')
from util.misc import nested_tensor_from_tensor_list

from torch import nn, Tensor
from torchvision.ops._utils import _cat
from typing import Optional, List, Tuple
from torchvision.ops.boxes import batched_nms, box_iou
from torchvision.models.detection import transform

import pocket.models as models

from transforms import HOINetworkTransform
from interaction_head import InteractionHead

class GenericHOIDetector(nn.Module):
    """A generic architecture for HOI detector

    Parameters:
    -----------
        detector: nn.Module
        interaction_head: nn.Module
    """
    def __init__(self,
        detector: nn.Module, criterion: nn.Module,
        postprocessors: nn.Module, interaction_head: nn.Module,
        box_score_thresh: float, fg_iou_thresh: float,
        # Dataset parameters
        human_idx: int, num_classes: int,
        # Training parameters
        alpha: float = 0.5, gamma: float = 2.0,
        min_h_instances: int = 3, max_h_instances: int = 15,
        min_o_instances: int = 3, max_o_instances: int = 15
    ) -> None:
        super().__init__()
        self.detector = detector
        self.criterion = criterion
        self.postprocessors = postprocessors

        self.interaction_head = interaction_head

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.min_h_instances = min_h_instances
        self.max_h_instances = max_h_instances
        self.min_o_instances = min_o_instances
        self.max_o_instances = max_o_instances

    def generate_object_targets(self, targets, nms_thresh=0.7):
        object_targets = []
        for target in targets:
            boxes = torch.cat([
                target['boxes_h'],
                target['boxes_o']
            ])
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            boxes[:, :2] -= 1
            labels = torch.cat([
                49 * torch.ones_like(target['object']),
                target['object']
            ])
            # Remove overlapping ground truth boxes
            keep = batched_nms(
                boxes, torch.ones(len(boxes)),
                labels, iou_threshold=nms_thresh
            )
            boxes = boxes[keep]
            labels = labels[keep]
            object_targets.append(dict(
                boxes=boxes, labels=labels
            ))
        return object_targets

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, targets["boxes_h"]),
            box_iou(boxes_o, targets["boxes_o"])
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets["labels"][y]] = 1

        return labels

    def compute_detection_loss(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return losses

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_cross_entropy(
            torch.log(
                (prior + 1e-8) / (1 + torch.exp(-logits) - prior)
            ), labels, reduction='sum'
        )

        return loss / n_p

    def prepare_region_proposals(self, boxes, scores, labels, hidden_states):
        results = []
        for bx, sc, lb, hs in zip(boxes, scores, labels, hidden_states):
            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)
            if self.training:
                is_human = lb == self.human_idx
                hum = torch.nonzero(is_human).squeeze(1)
                obj = torch.nonzero(is_human == 0).squeeze(1)
                n_human = len(hum); n_object = len(obj)
                # Keep the number of human and object instances in a specified interval
                if n_human < self.min_h_instances:
                    keep_h = sc[hum].argsort(descending=True)[:self.min_h_instances]
                    keep_h = hum[keep_h]
                elif n_human > self.max_h_instances:
                    keep_h = sc[hum].argsort(descending=True)[:self.max_h_instances]
                    keep_h = hum[keep_h]
                else:
                    keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                    keep_h = keep[keep_h]

                if n_object < self.min_o_instances:
                    keep_o = sc[obj].argsort(descending=True)[:self.min_o_instances]
                    keep_o = obj[keep_o]
                elif n_object > self.max_o_instances:
                    keep_o = sc[obj].argsort(descending=True)[:self.max_o_instances]
                    keep_o = obj[keep_o]
                else:
                    keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                    keep_o = keep[keep_o]

                keep = torch.cat([keep_h, keep_o])

            results.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                hidden_states=hs[keep]
            ))

        return results

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, attn_maps):
        n = [len(b) for b in boxes]
        logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, attn in zip(
            boxes, bh, bo, logits, prior, objects, attn_maps
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr)
            scores = torch.sigmoid(lg[x, y])
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * prior[x, y], objects=obj[y],
                attn_maps=attn
            ))

        return detections

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
            images: List[Tensor]
            targets: List[dict]

        Returns:
        --------
            results: List[dict]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)
        object_targets = self.generate_object_targets(targets)

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        features, pos = self.detector.backbone(images)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])[0]

        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()

        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.detector.aux_loss:
            results['aux_outputs'] = self.detector._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            detection_loss = self.compute_detection_loss(results, object_targets)

        scores, labels, boxes = self.postprocessors(results, image_sizes)
        region_props = self.prepare_region_proposals(boxes, scores, labels, hs)

        logits, prior, bh, bo, objects, attn_maps = self.interaction_head(
            features, image_sizes, region_props
        )
        boxes = [r['boxes'] for r in region_props]

        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)
            loss_dict = dict(
                detection_loss=detection_loss,
                interaction_loss=interaction_loss
            )
            return loss_dict

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, attn_maps)
        return detections

# class SpatiallyConditionedGraph(GenericHOINetwork):
#     def __init__(self,
#         object_to_action: List[list],
#         human_idx: int,
#         # Backbone parameters
#         backbone_name: str = "resnet50",
#         pretrained: bool = True,
#         # Pooler parameters
#         output_size: int = 7,
#         sampling_ratio: int = 2,
#         # Box pair head parameters
#         node_encoding_size: int = 1024,
#         representation_size: int = 1024,
#         num_classes: int = 117,
#         box_score_thresh: float = 0.2,
#         fg_iou_thresh: float = 0.5,
#         num_iterations: int = 2,
#         distributed: bool = False,
#         # Transformation parameters
#         min_size: int = 800, max_size: int = 1333,
#         image_mean: Optional[List[float]] = None,
#         image_std: Optional[List[float]] = None,
#         postprocess: bool = True,
#         # Preprocessing parameters
#         box_nms_thresh: float = 0.5,
#         max_human: int = 15,
#         max_object: int = 15
#     ) -> None:

#         detector = models.fasterrcnn_resnet_fpn(backbone_name,
#             pretrained=pretrained)
#         backbone = detector.backbone

#         box_roi_pool = MultiScaleRoIAlign(
#             featmap_names=['0', '1', '2', '3'],
#             output_size=output_size,
#             sampling_ratio=sampling_ratio
#         )

#         box_pair_predictor = nn.Linear(representation_size * 2, num_classes)

#         interaction_head = InteractionHead(
#             box_roi_pool=box_roi_pool,
#             box_pair_predictor=box_pair_predictor,
#             out_channels=backbone.out_channels,
#             roi_pool_size=output_size,
#             node_encoding_size=node_encoding_size,
#             representation_size=representation_size,
#             human_idx=human_idx,
#             num_classes=num_classes,
#             object_class_to_target_class=object_to_action,
#             num_iter=num_iterations,
#             box_nms_thresh=box_nms_thresh,
#             box_score_thresh=box_score_thresh,
#             fg_iou_thresh=fg_iou_thresh,
#             max_human=max_human,
#             max_object=max_object,
#             distributed=distributed
#         )

#         if image_mean is None:
#             image_mean = [0.485, 0.456, 0.406]
#         if image_std is None:
#             image_std = [0.229, 0.224, 0.225]
#         transform = HOINetworkTransform(min_size, max_size,
#             image_mean, image_std)

#         super().__init__(backbone, interaction_head, transform, postprocess)

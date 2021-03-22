"""
Models

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torchvision.ops.boxes as box_ops

from torch import nn, Tensor
from torchvision.ops._utils import _cat
from typing import Optional, List, Tuple
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import transform

import pocket.models as models

from transforms import HOINetworkTransform
from interaction_head import InteractionHead, GraphHead

class GenericHOINetwork(nn.Module):
    """A generic architecture for HOI classification

    Parameters:
    -----------
        backbone: nn.Module
        interaction_head: nn.Module
        transform: nn.Module
        postprocess: bool
            If True, rescale bounding boxes to original image size
    """
    def __init__(self,
        backbone: nn.Module, interaction_head: nn.Module,
        transform: nn.Module, postprocess: bool = True
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.interaction_head = interaction_head
        self.transform = transform

        self.postprocess = postprocess

    def preprocess(self,
        images: List[Tensor],
        detections: List[dict],
        targets: Optional[List[dict]] = None
    ) -> Tuple[
        List[Tensor], List[dict],
        List[dict], List[Tuple[int, int]]
    ]:
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        for det, o_im_s, im_s in zip(
            detections, original_image_sizes, images.image_sizes
        ):
            boxes = det['boxes']
            boxes = transform.resize_boxes(boxes, o_im_s, im_s)
            det['boxes'] = boxes

        return images, detections, targets, original_image_sizes

    def forward(self,
        images: List[Tensor],
        detections: List[dict],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
            images: List[Tensor]
            detections: List[dict]
            targets: List[dict]

        Returns:
        --------
            results: List[dict]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images, detections, targets, original_image_sizes = self.preprocess(
                images, detections, targets)

        features = self.backbone(images.tensors)
        results = self.interaction_head(features, detections, 
            images.image_sizes, targets)

        if self.postprocess and results is not None:
            return self.transform.postprocess(
                results,
                images.image_sizes,
                original_image_sizes
            )
        else:
            return results

class SpatioAttentiveGraph(GenericHOINetwork):
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
            box_score_thresh=0.2,
            gamma=0.5,
            fg_iou_thresh=0.5,
            num_iterations=1,
            distributed=False,
            # Transformation parameters
            min_size=800, max_size=1333,
            image_mean=None, image_std=None,
            postprocess=True,
            # Preprocessing parameters
            box_nms_thresh=0.5,
            max_human=15,
            max_object=15
            ):

        detector = models.fasterrcnn_resnet_fpn(backbone_name,
            pretrained=pretrained)
        backbone = detector.backbone

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=output_size,
            sampling_ratio=sampling_ratio
        )

        box_pair_head = GraphHead(
            out_channels=backbone.out_channels,
            roi_pool_size=output_size,
            node_encoding_size=node_encoding_size,
            representation_size=representation_size,
            num_cls=num_classes,
            human_idx=human_idx,
            object_class_to_target_class=object_to_action,
            fg_iou_thresh=fg_iou_thresh,
            num_iter=num_iterations
        )

        box_pair_predictor = nn.Linear(representation_size * 2, num_classes)
        box_pair_suppressor = nn.Linear(representation_size * 2, 1)

        interaction_head = InteractionHead(
            box_roi_pool=box_roi_pool,
            box_pair_head=box_pair_head,
            box_pair_suppressor=box_pair_suppressor,
            box_pair_predictor=box_pair_predictor,
            num_classes=num_classes,
            human_idx=human_idx,
            box_nms_thresh=box_nms_thresh,
            box_score_thresh=box_score_thresh,
            max_human=max_human,
            max_object=max_object,
            distributed=distributed
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = HOINetworkTransform(min_size, max_size,
            image_mean, image_std)

        super().__init__(backbone, interaction_head, transform, postprocess)

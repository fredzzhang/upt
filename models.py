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

from transforms import HOINetworkTransform
from interaction_head import InteractionHead, GraphHead

class GenericHOINetwork(nn.Module):
    """A generic architecture for HOI classification

    Arguments:
        backbone(nn.Module)
        interaction_head(nn.Module)
        transform(nn.Module)
        postprocess(bool): If True, rescale bounding boxes to original image size
    """
    def __init__(self, backbone, interaction_head, transform, postprocess=True):
        super().__init__()
        self.backbone = backbone
        self.interaction_head = interaction_head
        self.transform = transform

        self.postprocess = postprocess

    def preprocess(self, images, detections, targets=None):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        for det, o_im_s, im_s in zip(
            detections, original_image_sizes, images.image_sizes
        ):
            boxes = det['boxes']
            boxes = transform.resize_boxes(boxes, o_im_s, im_s)
            det['boxes'] = boxes

        return images, detections, targets, original_image_sizes

    def forward(self, images, detections, targets=None):
        """
        Arguments:
            images(list[Tensor])
            detections(list[dict])
            targets(list[dict])
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

class BoxPairPredictor(nn.Module):
    def __init__(self, input_size, representation_size, num_classes):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, num_classes)
        )
    def forward(self, x):
        return self.predictor(x)

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
            fg_iou_thresh=0.5,
            num_iterations=1,
            # Transformation parameters
            min_size=800, max_size=1333,
            image_mean=None, image_std=None,
            postprocess=True,
            # Preprocessing parameters
            box_nms_thresh=0.5,
            max_human=15,
            max_object=15
            ):

        backbone = models.fasterrcnn_resnet_fpn(backbone_name,
            pretrained=pretrained).backbone

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

        interaction_head = InteractionHead(
            box_roi_pool=box_roi_pool,
            box_pair_head=box_pair_head,
            box_pair_predictor=box_pair_predictor,
            num_classes=num_classes,
            human_idx=human_idx,
            box_nms_thresh=box_nms_thresh,
            max_human=max_human,
            max_object=max_object
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = HOINetworkTransform(min_size, max_size,
            image_mean, image_std)

        super().__init__(backbone, interaction_head, transform, postprocess)

    def state_dict(self):
        """Override method to only return state dict of the interaction head"""
        return self.interaction_head.state_dict()
    def load_state_dict(self, x):
        """Override method to only load state dict of the interaction head"""
        self.interaction_head.load_state_dict(x)

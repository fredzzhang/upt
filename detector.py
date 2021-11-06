"""
Human-object interaction detector

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch


from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms

from ops import SetCriterion, box_cxcywh_to_xyxy
from interaction_head import InteractionHead

import sys
sys.path.append('detr')
from models import build_model
from util.misc import nested_tensor_from_tensor_list

class GenericHOIDetector(nn.Module):
    """A generic architecture for HOI detector

    Parameters:
    -----------
        detector: nn.Module
        interaction_head: nn.Module
    """
    def __init__(self,
        detector: nn.Module, criterion: nn.Module,
        interaction_head: nn.Module,
        verb_predictor: nn.Module,
        # Dataset parameters
        human_idx: int, num_classes: int,
        # Training parameters
        alpha: float = .5, gamma: float = 2.,
        box_score_thresh: float = .2,
        min_instances: int = 3,
        max_instances: int = 15,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.criterion = criterion

        self.interaction_head = interaction_head
        self.verb_predictor = verb_predictor

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances

    def sampler(self, scores, labels):
        keep = torch.nonzero(scores >= self.box_score_thresh).squeeze(1)

        is_hum = labels.eq(self.human_idx)
        hum_idx = torch.nonzero(is_hum).squeeze(1)
        obj_idx = torch.nonzero(is_hum == 0).squeeze(1)
        n_hum = is_hum[keep].sum(); n_obj = len(keep) - n_hum

        # Sample human and object instances
        if n_hum < self.min_instances:
            keep_h = scores[hum_idx].argsort(descending=True)[:self.min_instances]
            keep_h = hum_idx[keep_h]
        elif n_hum > self.max_instances:
            keep_h = scores[hum_idx].argsort(descending=True)[:self.max_instances]
            keep_h = hum_idx[keep_h]
        else:
            keep_h = torch.nonzero(is_hum[keep]).squeeze(1)
            keep_h = keep[keep_h]

        if n_obj < self.min_instances:
            keep_o = scores[obj_idx].argsort(descending=True)[:self.min_instances]
            keep_o = obj_idx[keep_o]
        elif n_obj > self.max_instances:
            keep_o = scores[obj_idx].argsort(descending=True)[:self.max_instances]
            keep_o = obj_idx[keep_o]
        else:
            keep_o = torch.nonzero(is_hum[keep] == 0).squeeze(1)
            keep_o = keep[keep_o]

        keep = torch.cat([keep_h, keep_o])
        return keep

    def prepare_region_proposals(self, results, hidden_states):
        logits, boxes = results['pred_logits'], results['pred_boxes']
        prob = torch.softmax(logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        region_props = []
        for bx, sc, lb, hs in zip(boxes, scores, labels, hidden_states):

            # Perform NMS
            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            # TODO Maybe take this dimension from an argument
            hs = hs[keep].view(-1, 256)

            keep = self.sampler(sc, lb)
            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                hidden_states=hs[keep]
            ))

        return region_props

    def recover_boxes(self, boxes, image_size):
        h, w = image_size
        boxes = box_cxcywh_to_xyxy(boxes)
        scale_fct = torch.stack([w, h, w, h]).view(1, 4)
        return boxes * scale_fct

    @torch.no_grad()
    def postprocessing(self,
        boxes, idx_h, idx_o,
        logits, prior, objects,
        attn_maps, image_sizes
    ):
        n = [len(i) for i in idx_h]
        logits = logits.split(n)

        detections = []
        for bx, ih, io, lg, pr, obj, attn, size in zip(
            boxes, idx_h, idx_o, logits, prior, objects, attn_maps, image_sizes
        ):
            # Recover the unary boxes before regression
            bx = self.recover_boxes(bx, size)

            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            detections.append(dict(
                boxes=bx, pairing=torch.stack([ih[x], io[x]]),
                scores=scores * pr[x, y], repeat=x, labels=y,
                objects=obj, attn_maps=attn
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

        # if self.training:
        #     object_targets = self.generate_object_targets(targets)
        #     detection_loss = self.compute_detection_loss(results, object_targets)

        region_props = self.prepare_region_proposals(results, hs[-1])

        pairwise_features, prior, idx_h, idx_o, objects, attn_maps = self.interaction_head(
            features[-1].tensors, region_props
        )

        pairwise_features = torch.cat(pairwise_features)
        logits = self.verb_predictor(pairwise_features)

        boxes = [r['boxes'] for r in region_props]

        if self.training:
            loss_dict = self.criterion(boxes, idx_h, idx_o, objects, prior, logits, None, targets)
            return loss_dict

        detections = self.postprocessing(boxes, idx_h, idx_o, logits, prior, objects, attn_maps, image_sizes)
        return detections

def build_detector(args, class_corr):
    detr, _, _ = build_model(args)
    if os.path.exists(args.pretrained):
        print(f"Load pre-trained model from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained)['model_state_dict'])

    verb_predictor = torch.nn.Linear(args.repr_dim * 2, args.num_classes)

    interaction_head = InteractionHead(
        args.hidden_dim, args.repr_dim,
        detr.backbone[0].num_channels,
        args.num_classes, args.human_idx, class_corr
    )
    criterion = SetCriterion(args)
    detector = GenericHOIDetector(
        detr, criterion, interaction_head,
        verb_predictor, None,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
    )
    return detector

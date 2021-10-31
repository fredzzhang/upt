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

from ops import (
    SetCriterion, BoxPairCoder, BalancedBoxSampler,
    box_cxcywh_to_xyxy
)
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
        detector: nn.Module, criterion: nn.Module, interaction_head: nn.Module,
        verb_predictor: nn.Module, bbox_regressor: nn.Module,
        # Dataset parameters
        human_idx: int, num_classes: int,
        # Training parameters
        alpha: float = .5, gamma: float = 2.,
        box_score_thresh: float = .2, high_conf_perc: float = .8,
        n_h_instances: int = 10, n_o_instances: int = 15,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.criterion = criterion

        self.interaction_head = interaction_head
        self.verb_predictor = verb_predictor
        self.bbox_regressor = bbox_regressor

        self.sampler = BalancedBoxSampler(
            threshold=box_score_thresh,
            perc=high_conf_perc
        )

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.n_h_instances = n_h_instances
        self.n_o_instances = n_o_instances

        self.box_pair_coder = BoxPairCoder()

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

            h_idx = torch.nonzero(lb == self.human_idx).squeeze(1)
            o_idx = torch.nonzero(lb != self.human_idx).squeeze(1)

            if self.training:
                # Sample a fixed number of human and object boxes
                keep_h = torch.cat(self.sampler(sc[h_idx], self.n_h_instances))
                keep_o = torch.cat(self.sampler(sc[o_idx], self.n_o_instances))
            else:
                keep_h = torch.argsort(sc[h_idx], descending=True)[:self.n_h_instances]
                keep_o = torch.argsort(sc[o_idx], descending=True)[:self.n_o_instances]
            keep = torch.cat([h_idx[keep_h], o_idx[keep_o]])

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
        boxes, idx_h, idx_o, bbox_deltas,
        logits, prior, objects, attn_maps, image_sizes
    ):
        n = [len(i) for i in idx_h]
        logits = logits.split(n)
        # bbox_deltas = bbox_deltas.split(n)

        detections = []
        for bx, ih, io, lg, pr, obj, attn, size in zip(
            boxes, idx_h, idx_o, logits, prior, objects, attn_maps, image_sizes
        ):
            # Recover the unary boxes before regression
            bx = self.recover_boxes(bx, size)

            # Recover the regressed box pairs
            # bx_h_post, bx_o_post = self.box_pair_coder.decode(bx[idx_h], bx[idx_o], delta)
            # bx_h_post = self.recover_boxes(bx_h_post, size)
            # bx_o_post = self.recover_boxes(bx_o_post, size)

            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            detections.append(dict(
                boxes=bx, pairing=torch.stack([ih[x], io[x]]),
                # boxes_h=bx_h_post, boxes_o=bx_o_post,
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

        # results = self.postprocessors(results, image_sizes)
        region_props = self.prepare_region_proposals(results, hs[-1])

        pairwise_features, prior, idx_h, idx_o, objects, attn_maps = self.interaction_head(
            features[-1].tensors, region_props
        )

        pairwise_features = torch.cat(pairwise_features)
        logits = self.verb_predictor(pairwise_features)
        # bbox_deltas = self.bbox_regressor(pairwise_features)

        boxes = [r['boxes'] for r in region_props]

        if self.training:
            loss_dict = self.criterion(boxes, idx_h, idx_o, objects, prior, logits, None, targets)
            # loss_dict = dict(
            #     detection_loss=detection_loss,
            #     interaction_loss=interaction_loss
            # )
            return loss_dict

        detections = self.postprocessing(boxes, idx_h, idx_o, None, logits, prior, objects, attn_maps, image_sizes)
        return detections

def build_detector(args, class_corr):
    detr, _, _ = build_model(args)
    if os.path.exists(args.pretrained):
        print(f"Load pre-trained model from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained)['model_state_dict'])

    verb_predictor = torch.nn.Linear(args.repr_dim * 2, args.num_classes)
    # bbox_regressor = torch.nn.Linear(args.repr_dim * 2, 8)

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
        high_conf_perc=args.high_conf_perc,
        n_h_instances=args.num_humans,
        n_o_instances=args.num_objects
    )
    return detector

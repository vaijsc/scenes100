import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from models.detector import Detector, SetCriterion, PostProcess
from models.matcher import HungarianMatcher
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


__all__ = ["YOLOS"]

@META_ARCH_REGISTRY.register()
class YOLOS(nn.Module):
    """
    YOLOS wrapper for detectron
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_classes = cfg.MODEL.YOLOS.NUM_CLASSES

        self.model = Detector(
            num_classes=cfg.MODEL.YOLOS.NUM_CLASSES,
            det_token_num=cfg.MODEL.YOLOS.DET_TOKEN_NUM,
            backbone_name=cfg.MODEL.YOLOS.BACKBONE_NAME,
            init_pe_size=cfg.MODEL.YOLOS.INIT_PE_SIZE,
            mid_pe_size=cfg.MODEL.YOLOS.MID_PE_SIZE
        )
        self.model.to(self.device)

        matcher = HungarianMatcher(cost_class=cfg.MODEL.MATCHER.SET_COST_CLASS, cost_bbox=cfg.MODEL.MATCHER.SET_COST_BBOX, cost_giou=cfg.MODEL.MATCHER.SET_COST_GIOU)
        weight_dict = {'loss_ce': 1, 'loss_bbox': cfg.MODEL.LOSS.BBOX_LOSS_COEF}
        weight_dict['loss_giou'] = cfg.MODEL.LOSS.GIOU_LOSS_COEF

        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(cfg.MODEL.YOLOS.NUM_CLASSES, matcher=matcher, weight_dict=weight_dict,
                                eos_coef=cfg.MODEL.LOSS.EOS_COEF, losses=losses)

        self.criterion.to(self.device)
        self.postprocessors = {'bbox': PostProcess()}

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.model(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results   
    
    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

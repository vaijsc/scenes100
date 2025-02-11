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

from models.dino.dino import DINO, SetCriterion, PostProcess, build_dino
from models.dino.matcher import HungarianMatcher
from models.dino.backbone import Joiner
from models.dino.position_encoding import PositionEmbeddingSineHW
from models.dino.deformable_transformer import build_deformable_transformer 
from models.dino.matcher import build_matcher
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.misc import NestedTensor
from cfg_to_args import cfg_to_args
import copy

__all__ = ["Dino"]

class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        # self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels
        self.num_channels = [backbone_shape[key].channels for key in list(backbone_shape.keys())]

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class Dino(nn.Module):
    """
    DINO wrapper for detectron
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        args = cfg_to_args(cfg)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        
        num_classes = args.num_classes
        device = torch.device(args.device)

        N_steps = args.hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSineHW(N_steps, args.pe_temperatureH, args.pe_temperatureW, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        transformer = build_deformable_transformer(args)

        try:
            match_unstable_error = args.match_unstable_error
            dn_labelbook_size = args.dn_labelbook_size
        except:
            match_unstable_error = True
            dn_labelbook_size = num_classes

        try:
            dec_pred_class_embed_share = args.dec_pred_class_embed_share
        except:
            dec_pred_class_embed_share = True
        try:
            dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
        except:
            dec_pred_bbox_embed_share = True

        model = DINO(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=True,
            iter_update=True,
            query_dim=4,
            random_refpoints_xy=args.random_refpoints_xy,
            fix_refpoints_hw=args.fix_refpoints_hw,
            num_feature_levels=args.num_feature_levels,
            nheads=args.nheads,
            dec_pred_class_embed_share=dec_pred_class_embed_share,
            dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
            # two stage
            two_stage_type=args.two_stage_type,
            # box_share
            two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
            two_stage_class_embed_share=args.two_stage_class_embed_share,
            decoder_sa_type=args.decoder_sa_type,
            num_patterns=args.num_patterns,
            dn_number = args.dn_number if args.use_dn else 0,
            dn_box_noise_scale = args.dn_box_noise_scale,
            dn_label_noise_ratio = args.dn_label_noise_ratio,
            dn_labelbook_size = dn_labelbook_size,
        )
        if args.masks:
            model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        matcher = build_matcher(args)

        # prepare weight dict
        weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef
        clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

        
        # for DN training
        if args.use_dn:
            weight_dict['loss_ce_dn'] = args.cls_loss_coef
            weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
            weight_dict['loss_giou_dn'] = args.giou_loss_coef

        if args.masks:
            weight_dict["loss_mask"] = args.mask_loss_coef
            weight_dict["loss_dice"] = args.dice_loss_coef
        clean_weight_dict = copy.deepcopy(weight_dict)

        # TODO this is a hack
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if args.two_stage_type != 'no':
            interm_weight_dict = {}
            try:
                no_interm_box_loss = args.no_interm_box_loss
            except:
                no_interm_box_loss = False
            _coeff_weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
                'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            }
            try:
                interm_loss_coef = args.interm_loss_coef
            except:
                interm_loss_coef = 1.0
            interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
            weight_dict.update(interm_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']
        if args.masks:
            losses += ["masks"]
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                focal_alpha=args.focal_alpha, losses=losses,
                                )
        criterion.to(device)
        postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
        if args.masks:
            postprocessors['segm'] = PostProcessSegm()
            if args.dataset_file == "coco_panoptic":
                is_thing_map = {i: i <= 90 for i in range(201)}
                postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

        model.to(self.device)
        criterion.to(self.device)

        self.model = model
        self.criterion = criterion
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
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)
        else:
            targets = None
        output = self.model(images, targets)

        if self.training:
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            # targets = self.prepare_targets(gt_instances)
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

        N, _, H, W = images.tensor.shape
        mask = torch.ones((N, H, W), dtype=torch.bool, device=self.device)
        for img_idx, (h, w) in enumerate(images.image_sizes):
            mask[img_idx, : h, : w] = 0
        images.mask = mask

        return images

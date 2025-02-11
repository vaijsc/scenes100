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

        model, criterion, postprocessors = build_dino(args)

        model.to(self.device)
        criterion.to(self.device)

        self.model = model
        self.criterion = criterion
        self.to(self.device)
        self.num_select = args.num_select

        self.budget = cfg.MODEL.ADAPTIVE_BUDGET
        self.video_id_to_index = {}
        self.used_indices = {}
        self.un_used_indices = {i: True for i in range(0, self.budget)}

    def get_training_assignment(self, batched_inputs):
        if self.training:
            for im in batched_inputs:
                if im['video_id'] != 'coco' and im['video_id'] not in self.video_id_to_index:
                    if len(self.un_used_indices) > 0:
                        i = sorted(list(self.un_used_indices.keys()))[0]
                        self.video_id_to_index[im['video_id']] = i
                        del self.un_used_indices[i]
                        self.used_indices[i] = True
                    else:
                        self.video_id_to_index[im['video_id']] = np.random.choice(list(self.used_indices.keys()))
            module_indices = []
            for im in batched_inputs:
                # randomly train 1 path for COCO images
                if im['video_id'] == 'coco':
                    if len(self.used_indices) > 0:
                        module_indices.append(np.random.choice(list(self.used_indices.keys())))
                    else:
                        module_indices.append(np.random.choice(list(self.un_used_indices.keys())))
                else:
                    module_indices.append(self.video_id_to_index[im['video_id']])
        else:
            module_indices = []
            for im in batched_inputs:
                # at inference time, assign first module for all unseen video IDs
                if 'video_id' not in im:
                    module_indices.append(0)
                    continue
                if im['video_id'] == 'coco' or im['video_id'] not in self.video_id_to_index:
                    module_indices.append(0)
                else:
                    module_indices.append(self.video_id_to_index[im['video_id']])


        return batched_inputs, module_indices

    def forward(self, batched_inputs, return_features=False):
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
        batched_inputs, module_indices = self.get_training_assignment(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)
        else:
            targets = None
        
        output = self.model(images, targets, module_indices, return_features)
        if return_features:
            return output
        # breakpoint()
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
                ratio = input_per_image['image'].shape[1] / height
                # breakpoint()
                # self.draw(input_per_image['image']/255, r, ratio=ratio, file_name=f'test.png')
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

        num_select = self.num_select
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // box_cls.shape[2]
        labels = topk_indexes % box_cls.shape[2]
        box_pred = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        results = []
        # # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            # threshold = 0.3
            # select_mask = scores_per_image > threshold
            # scores_per_image = scores_per_image[select_mask]
            # labels_per_image = labels_per_image[select_mask]
            # box_pred_per_image = box_pred_per_image[select_mask]

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
        # images.mask = mask
        nested_tensor = NestedTensor(images.tensor, mask)
        nested_tensor.image_sizes = images.image_sizes
        return nested_tensor
    
    @staticmethod
    def draw(image_tensor, detection_labels, ratio=None, file_name="test_preprocess.png"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        # Convert the PyTorch tensor to a NumPy array for visualization
        image_tensor = image_tensor.cpu()
        image_np = image_tensor.permute(1, 2, 0).numpy()
        
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image_np)

        # Process the detection labels and draw bounding boxes
        assert ratio is not None, "ratio unavailable"
        detection_labels = detection_labels.pred_boxes.tensor.cpu()
        for detection in detection_labels:
            x1, y1, x2, y2 = detection
            x1 *= ratio
            x2 *= ratio
            y1 *= ratio
            y2 *= ratio

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.savefig(file_name)
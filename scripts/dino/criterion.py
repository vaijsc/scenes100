# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import os
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast


groundingdino_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'GroundingDINO')
sys.path.append(groundingdino_dir)
from groundingdino.util import box_ops
from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss
from groundingdino.util.misc import get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    def __init__(self, matcher, focal_alpha, focal_gamma, losses):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        # self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_classes = 1 # class-agnostic

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        # with torch.no_grad():
        #     losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #     losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def token_sigmoid_binary_focal_loss(self, outputs, targets, indices, num_boxes):
        pred_logits=outputs['pred_logits']
        new_targets=outputs['one_hot'].to(pred_logits.device)
        text_mask=outputs['text_mask']

        assert (new_targets.dim() == 3)
        assert (pred_logits.dim() == 3)  # batch x from x to

        bs, n, _ = pred_logits.shape
        alpha=self.focal_alpha
        gamma=self.focal_gamma
        if text_mask is not None:
            # ODVG: each sample has different mask
            text_mask = text_mask.repeat(1, pred_logits.size(1)).view(outputs['text_mask'].shape[0],-1,outputs['text_mask'].shape[1])
            pred_logits = torch.masked_select(pred_logits, text_mask)
            new_targets = torch.masked_select(new_targets, text_mask)

        new_targets=new_targets.float()
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, new_targets, reduction="none")
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * new_targets + (1 - alpha) * (1 - new_targets)
            loss = alpha_t * loss

        total_num_pos=0
        for batch_indices in indices:
            total_num_pos += len(batch_indices[0])
        num_pos_avg_per_gpu = max(total_num_pos , 1.0)
        loss = loss.sum() / num_pos_avg_per_gpu
        losses = {'loss_ce': loss}
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            # 'labels': self.token_sigmoid_binary_focal_loss,
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        device = next(iter(outputs.values())).device
        one_hot = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64) # torch.Size([bs, 900, 256])
        indices = self.matcher(outputs, targets)
        # token = outputs['token']

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # # label_map_list = []
        # # indices = []
        # # for j in range(len(cat_list)): # bs
        # #     label_map=[]
        # #     for i in range(len(cat_list[j])):
        # #         label_id=torch.tensor([i])
        # #         per_label=create_positive_map(token[j], label_id, cat_list[j], caption[j])
        # #         label_map.append(per_label)
        # #     label_map=torch.stack(label_map,dim=0).squeeze(1)
        # #     label_map_list.append(label_map)
        # # for j in range(len(cat_list)): # bs
        # #     for_match = {
        # #         "pred_logits" : outputs['pred_logits'][j].unsqueeze(0),
        # #         "pred_boxes" : outputs['pred_boxes'][j].unsqueeze(0)
        # #     }
        # #     inds = self.matcher(for_match, [targets[j]], label_map_list[j])
        # #     indices.extend(inds)
        # # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
        # # - index_i is the indices of the selected predictions (in order)
        # # - index_j is the indices of the corresponding selected targets (in order)

        # tgt_ids = [v["labels"].cpu() for v in targets] # len(tgt_ids) == bs
        # for i in range(len(indices)):
        #     tgt_ids[i] = tgt_ids[i][indices[i][1]]
        #     one_hot[i, indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
        # outputs['one_hot'] = one_hot
        # # if return_indices:
        # #     indices0_copy = indices
        # #     indices_list = []

        # # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes_list = [len(t["labels"]) for t in targets]
        # num_boxes = sum(num_boxes_list)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if 'aux_outputs' in outputs:
        #     for idx, aux_outputs in enumerate(outputs['aux_outputs']):
        #         indices = []
        #         for j in range(len(cat_list)): # bs
        #             aux_output_single = {
        #                 'pred_logits' : aux_outputs['pred_logits'][j].unsqueeze(0),
        #                 'pred_boxes': aux_outputs['pred_boxes'][j].unsqueeze(0)
        #             }
        #             inds = self.matcher(aux_output_single, [targets[j]], label_map_list[j])
        #             indices.extend(inds)
        #         one_hot_aux = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64)
        #         tgt_ids = [v["labels"].cpu() for v in targets]
        #         for i in range(len(indices)):
        #             tgt_ids[i]=tgt_ids[i][indices[i][1]]
        #             one_hot_aux[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
        #         aux_outputs['one_hot'] = one_hot_aux
        #         aux_outputs['text_mask'] = outputs['text_mask']
        #         # if return_indices:
        #         #     indices_list.append(indices)
        #         for loss in self.losses:
        #             kwargs = {}
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
        #             l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
        #             losses.update(l_dict)

        # interm_outputs loss
        # if 'interm_outputs' in outputs:
        #     interm_outputs = outputs['interm_outputs']
        #     indices = []
        #     for j in range(len(cat_list)): # bs
        #         interm_output_single = {
        #             'pred_logits' : interm_outputs['pred_logits'][j].unsqueeze(0),
        #             'pred_boxes': interm_outputs['pred_boxes'][j].unsqueeze(0)
        #         }
        #         inds = self.matcher(interm_output_single, [targets[j]], label_map_list[j])
        #         indices.extend(inds)
        #     one_hot_aux = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64)
        #     tgt_ids = [v["labels"].cpu() for v in targets]
        #     for i in range(len(indices)):
        #         tgt_ids[i]=tgt_ids[i][indices[i][1]]
        #         one_hot_aux[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
        #     interm_outputs['one_hot'] = one_hot_aux
        #     interm_outputs['text_mask'] = outputs['text_mask']
        #     # if return_indices:
        #     #     indices_list.append(indices)
        #     for loss in self.losses:
        #         kwargs = {}
        #         l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
        #         l_dict = {k + f'_interm': v for k, v in l_dict.items()}
        #         losses.update(l_dict)

        # if return_indices:
        #     indices_list.append(indices0_copy)
        #     return losses, indices_list
        return losses

#!python3

import torch

from matcher import HungarianMatcher
from criterion import SetCriterion


# boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
if __name__ == '__main__':
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
    outputs = {
        'pred_logits': torch.tensor([[[0], [2], [10]]]).float(),
        'pred_boxes': torch.tensor([[
            [0.51, 0.51, 0.12, 0.11],
            [0.50, 0.53, 0.11, 0.10],
            [0.31, 0.72, 0.21, 0.20],
        ]]).float(),
    }
    targets = [{
        'labels': torch.tensor([0, 0]).long(),
        'boxes': torch.tensor([
            [0.3, 0.7, 0.2, 0.2],
            [0.5, 0.5, 0.1, 0.1],
        ]).float()
    }]
    for idx_pred, idx_gt in matcher(outputs, targets):
        print(idx_pred, idx_gt)

    loss_fn = SetCriterion(
        matcher     = matcher,
        focal_alpha = 0.25,
        focal_gamma = 2,
        losses      = ['labels', 'boxes']
    )
    print(loss_fn(outputs, targets))

'''
https://github.com/longzw1997/Open-GroundingDino/blob/main/config/cfg_coco.py

aux_loss = True
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
mask_loss_coef = 1.0
dice_loss_coef = 1.0
focal_alpha = 0.25
focal_gamma = 2.0
decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1
dec_pred_class_embed_share = True
match_unstable_error = True
use_detached_boxes_dec_out = False
dn_scalar = 100
'''

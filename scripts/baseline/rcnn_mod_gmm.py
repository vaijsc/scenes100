#!python3

import numpy as np
import enum
import sklearn.utils
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.events import get_event_storage
import detectron2.modeling
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats


class AnnotationType(enum.IntEnum):
    RELIABLE = 0
    PSEUDO   = 1


#############################################################
#####                       NOTE                        #####
##### It seems that if an Instances field has gradient, #####
##### and is indexed, and stored back to the instances, #####
##### the gradient backward will be cut.                #####
#############################################################

# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
# apply per-bbox weights on the losses
class GeneralizedRCNNFinetune(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        assert 'proposals' not in batched_inputs[0], 'pre-computed proposals not supported'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs] if 'instances' in batched_inputs[0] else None
        features = self.backbone(images.tensor)
        for i in range(0, len(gt_instances)):
            if gt_instances[i].annotation_type[0].item() == AnnotationType.PSEUDO:
                gt_instances[i].pseudo_scores = self.weight_estimator(images[i], gt_instances[i], self.backbone)

        # current_image['batch'] = [x['image'].detach().cpu().numpy() for x in batched_inputs]
        # _t = self.roi_heads.box_predictor.density_estimator_per_class[0].means
        # _tt = self.backbone.fpn_output5.weight.data
        # print(_t[0, 2].item(), _t[1, 1].item(), _tt[2, 0, 0, 0].item(), _tt[0, 0, 1, 0].item()) # check if gradient is here

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # print(losses)
        # losses['loss_cls'] = losses['loss_box_reg'] = losses['loss_rpn_cls'] = losses['loss_rpn_loc'] = torch.tensor(0.0)
        # print(losses)
        return losses

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.proposal_generator, detectron2.modeling.proposal_generator.rpn.RPN), 'rpn is not detectron2.modeling.proposal_generator.rpn.RPN'
        assert isinstance(net.roi_heads, detectron2.modeling.roi_heads.roi_heads.StandardROIHeads), 'roi is not detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'
        assert isinstance(net.roi_heads.box_predictor, detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers), 'roi is not detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers'

        net.__class__ = GeneralizedRCNNFinetune
        net.proposal_generator.__class__ = RPNFinetune
        net.roi_heads.__class__ = StandardROIHeadsFinetune
        net.roi_heads.box_predictor.__class__ = FastRCNNOutputLayersFinetune
        return net


# wrap detectron2/modeling/roi_heads/fast_rcnn.py:FastRCNNOutputLayers
class FastRCNNOutputLayersFinetune(detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers):
    def losses(self, predictions, proposals, targets=None, matched_gt_indices=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
        if not len(proposals):
            pseudo_scores = torch.empty(0)
        else:
            pseudo_scores = cat([targets[i].pseudo_scores[matched_gt_indices[i]] for i in range(0, len(targets))])
        # print(gt_classes.size(), gt_classes.min(), gt_classes.max(), pseudo_scores.size(), pseudo_scores.min(), pseudo_scores.max())
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, 'Proposals should not require gradients!'
            # If "gt_boxes" does not exist, the proposals must be all negative and should not be included in regression loss computation. Here we just use proposal_boxes as an arbitrary placeholder because its value won't be used in self.box_reg_loss().
            gt_boxes = cat([(p.gt_boxes if p.has('gt_boxes') else p.proposal_boxes).tensor for p in proposals], dim=0)
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        loss_cls = cross_entropy(scores, gt_classes, reduction='none')
        loss_cls = (loss_cls * pseudo_scores).mean() / pseudo_scores.mean()
        losses = {
            # 'loss_cls': cross_entropy(scores, gt_classes, reduction='mean'),
            'loss_cls': loss_cls,
            'loss_box_reg': self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes),
        }

        box_dim = proposal_boxes.size(1)
        assert proposal_deltas.size(1) != box_dim, 'does not supported cls-agnostic density estimation loss'
        N = len(targets)
        scales = [{'x': t.x_scale[0], 'y': t.y_scale[0]} for t in targets]
        gt_classes, scores = gt_classes.view(N, -1), scores.view(N, -1, scores.size(1))
        proposal_boxes, proposal_deltas = proposal_boxes.view(N, -1, proposal_boxes.size(1)), proposal_deltas.view(N, -1, proposal_deltas.size(1))
        gmm_log_likelihoods_classes = []
        for k in range(0, self.num_classes):
            pred_boxes, pred_boxes_norm, gmm_log_likelihoods = [None for _ in range(0, N)], [None for _ in range(0, N)], [None for _ in range(0, N)]
            for i in range(0, N):
                if int(targets[i].annotation_type[0].item()) == AnnotationType.RELIABLE:
                    continue
                # filter out all backgrounds
                fg_inds = detectron2.layers.nonzero_tuple((gt_classes[i] >= 0) & (gt_classes[i] < self.num_classes))[0]
                fg_pred_deltas = proposal_deltas[i].view(-1, self.num_classes, box_dim)[fg_inds]
                fg_proposal_boxes = proposal_boxes[i][fg_inds]
                fg_scores, fg_gt_classes = scores[i][fg_inds], gt_classes[i][fg_inds]

                # for each proposal, only apply class-specific loss when that class is predicted correctly
                fg_pred_class = fg_scores.argmax(dim=1)
                _mask = (fg_pred_class == fg_gt_classes) & (fg_pred_class == k)
                if _mask.sum() < 1:
                    continue
                pred_boxes[i] = self.box2box_transform.apply_deltas(fg_pred_deltas[:, k, :], fg_proposal_boxes)
                pred_boxes_norm[i] = pred_boxes[i] / torch.stack([scales[i]['x'], scales[i]['y'], scales[i]['x'], scales[i]['y']]).view(1, 4)
                gmm_log_likelihoods[i] = self.density_estimator_per_class[k](pred_boxes_norm[i])
            # helper_show(pred_boxes, gmm_log_likelihoods, 'ROIs class %d' % k)
            gmm_log_likelihoods_classes = gmm_log_likelihoods_classes + gmm_log_likelihoods
        gmm_log_likelihoods_classes = list(filter(lambda x: not x is None, gmm_log_likelihoods_classes))
        if len(gmm_log_likelihoods_classes) > 0:
            nlls = torch.cat(gmm_log_likelihoods_classes) * -1.0
            nlls = nlls - nlls.detach().min() # adding a constant should not affect gradient
            box_density_loss = nlls.mean() * self.density_estimator_per_class[k].weighted_std * self.density_loss_scale
        else:
            box_density_loss = torch.tensor(0.0).type(self.density_estimator_per_class[0].weighted_std.dtype)
        losses['loss_roi_density'] = box_density_loss

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


# wrap detectron2/modeling/roi_heads/roi_heads.py:StandardROIHeads
class StandardROIHeadsFinetune(detectron2.modeling.roi_heads.roi_heads.StandardROIHeads):
    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, '\'targets\' argument is required during training'
            proposals, matched_gt_indices = self.label_and_sample_proposals(proposals, targets)
        # del targets

        if self.training:
            losses = self._forward_box(features, proposals, targets, matched_gt_indices)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals, targets=None, matched_gt_indices=None):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, targets, matched_gt_indices)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes. In the case of learned proposals (e.g., RPN), when training starts the proposals will be low quality due to random initialization. It's possible that none of these initial proposals have high enough overlap with the gt objects to be used as positive examples for the second stage components (box head, cls head, mask head). Adding the gt boxes to the set of proposals ensures that the second stage components will have some positive examples from the start of training. For RPN, this augmentation improves convergence and empirically improves box AP on COCO by about 0.5 points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []
        matched_gt_indices = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets_per_image.gt_classes)

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            matched_gt_indices.append(matched_idxs[sampled_idxs])

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_" and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads like masks, keypoints, etc, will filter the proposals again, (by foreground/background, or number of keypoints in the image, etc) so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith('gt_') and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be. Therefore the returned proposals won't have any gt_* fields, except for a gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar('roi_head/num_fg_samples', np.mean(num_fg_samples))
        storage.put_scalar('roi_head/num_bg_samples', np.mean(num_bg_samples))

        return proposals_with_gt, matched_gt_indices


# wrap detectron2/modeling/proposal_generator/rpn.py:RPN
class RPNFinetune(detectron2.modeling.proposal_generator.rpn.RPN):
    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2).flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, 'RPN requires gt_instances in training!'
            gt_labels, gt_boxes, matched_gt_indices = self.label_and_sample_anchors(anchors, gt_instances)
            pseudo_scores = []
            for i in range(0, len(matched_gt_indices)):
                pseudo_scores.append(gt_instances[i].pseudo_scores[matched_gt_indices[i]])
            pseudo_annotation_mask = [int(gi.annotation_type[0].item()) for gi in gt_instances]
            scales = [{'x': gi.x_scale[0], 'y': gi.y_scale[0]} for gi in gt_instances]
            losses = self.losses(anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes, pseudo_scores, pseudo_annotation_mask, scales)
        else:
            losses = {}
        proposals = self.predict_proposals(anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes)
        return proposals, losses

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.
        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        matched_gt_indices = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_gt_indices.append(matched_idxs)
        return gt_labels, matched_gt_boxes, matched_gt_indices

    @torch.jit.unused
    def losses(self, anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes, pseudo_scores, pseudo_annotation_mask=None, scales=None):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.
        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        pseudo_scores = torch.stack(pseudo_scores)

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar('rpn/num_pos_anchors', num_pos_anchors / num_images)
        storage.put_scalar('rpn/num_neg_anchors', num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(anchors, self.box2box_transform, pred_anchor_deltas, gt_boxes, pos_mask, box_reg_loss_type=self.box_reg_loss_type, smooth_l1_beta=self.smooth_l1_beta)

        valid_mask = gt_labels >= 0
        # objectness_loss = F.binary_cross_entropy_with_logits(cat(pred_objectness_logits, dim=1)[valid_mask], gt_labels[valid_mask].to(torch.float32), reduction='sum')

        # apply bbox weights to objectness loss
        objectness_loss = F.binary_cross_entropy_with_logits(cat(pred_objectness_logits, dim=1)[valid_mask], gt_labels[valid_mask].to(torch.float32), reduction='none')
        pseudo_scores_valid = pseudo_scores[valid_mask]
        # print(objectness_loss.sum(), objectness_loss.size(), pseudo_scores_valid.size(), pseudo_scores_valid.min(), pseudo_scores_valid.max())
        objectness_loss = (objectness_loss * pseudo_scores_valid / pseudo_scores_valid.mean()).sum()
        normalizer = self.batch_size_per_image * num_images
        losses = {
            'loss_rpn_cls': objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer for loc loss. But it doesn't matter in practice
            'loss_rpn_loc': localization_loss / normalizer,
        }

        assert len(pseudo_annotation_mask) == len(scales) == pos_mask.size(0)
        N = len(pseudo_annotation_mask)
        if isinstance(anchors[0], Boxes):
            anchors_cat = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        else:
            anchors_cat = cat(anchors)
        pred_boxes, pred_boxes_norm, gmm_log_likelihoods = [None for _ in range(0, N)], [None for _ in range(0, N)], [None for _ in range(0, N)]
        for i in range(0, N):
            # only apply loss for pseudo labeled images
            if pseudo_annotation_mask[i] == AnnotationType.RELIABLE:
                continue
            pred_boxes[i] = self.box2box_transform.apply_deltas(cat([p[i] for p in pred_anchor_deltas], dim=0), anchors_cat)
            pred_boxes[i] = pred_boxes[i][pos_mask[i]]
            pred_boxes_norm[i] = pred_boxes[i] / torch.stack([scales[i]['x'], scales[i]['y'], scales[i]['x'], scales[i]['y']]).view(1, 4)
            gmm_log_likelihoods[i] = self.density_estimator_all(pred_boxes_norm[i])
        #     print(i, pred_boxes[i].size(), pred_boxes_norm[i].size(), gmm_log_likelihoods[i].size())
        # helper_show(pred_boxes, gmm_log_likelihoods, 'RPN proposals')
        gmm_log_likelihoods = list(filter(lambda x: not x is None, gmm_log_likelihoods))
        if len(gmm_log_likelihoods) > 0:
            nlls = torch.cat(gmm_log_likelihoods) * -1.0
            nlls = nlls - nlls.detach().min() # adding a constant should not affect gradient
            box_density_loss = nlls.mean() * self.density_estimator_all.weighted_std * self.density_loss_scale
        else:
            box_density_loss = torch.tensor(0.0).type(self.density_estimator_all.weighted_std.dtype)

        losses['loss_rpn_density'] = box_density_loss
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses


# estimate per-box weighting for pseudo annotated images
class WeightEstimator(nn.Module):
    def __init__(self, dummy=False):
        super(WeightEstimator, self).__init__()
        self.dummy = dummy

    def forward(self, image, instances, rcnn_fpn):
        if self.dummy:
            outputs = torch.ones_like(instances.gmm_likelihoods)
        else:
            outputs = instances.gmm_likelihoods
        # print(outputs)
        return outputs


class GMMDensityEstimator(nn.Module):
    '''
    differentiable GMM inference
    ported from slearn.mixture.GaussianMixture, copy learned parameters from a converged GMM object
    forward function calculates the log-likelihood of data samples
    '''

    data_type = torch.float32

    def __init__(self, gmm_sklearn : sklearn.mixture.GaussianMixture, dummy=False):
        super(GMMDensityEstimator, self).__init__()
        self.dummy = dummy
        if self.dummy:
            self.weighted_std = nn.Parameter(torch.tensor(1.0).type(self.data_type), requires_grad=True)
            print('use dummy GMMDensityEstimator')
            return

        assert hasattr(gmm_sklearn, 'converged_') and gmm_sklearn.converged_, 'GMM model not converged'
        assert gmm_sklearn.covariance_type == 'full', 'only support full covariance'
        self.n_components, self.d_features = gmm_sklearn.means_.shape
        self.weights, self.means, self.covars, self.precisions, self.precisions_cholesky = map(self.np_to_param, [
            gmm_sklearn.weights_, gmm_sklearn.means_, gmm_sklearn.covariances_, gmm_sklearn.precisions_, gmm_sklearn.precisions_cholesky_
        ])
        self.lower_bound = torch.tensor(gmm_sklearn.lower_bound_).type(self.data_type)
        self.pi = torch.tensor(np.pi).type(self.data_type)

        self.stds = np.zeros_like(gmm_sklearn.weights_)
        for i in range(0, self.n_components):
            eigvals = np.linalg.eigvals(gmm_sklearn.covariances_[i])
            self.stds[i] = np.absolute(eigvals).max() ** 0.5
        self.weighted_std = (self.stds * gmm_sklearn.weights_).sum()
        self.weighted_std = nn.Parameter(torch.tensor(self.weighted_std).type(self.data_type), requires_grad=True)
        print('GMMDensityEstimator: %d components, feature dimension %d, weighted std %.4f' % (self.n_components, self.d_features, self.weighted_std.item()))

    def forward(self, X):
        if self.dummy:
            log_likelihoods = torch.zeros((X.size(0),), dtype=self.data_type)
        else:
            N, D = X.size()
            assert D == self.d_features, 'input feature dimension mismatch'
            log_det = torch.sum(torch.log(self.precisions_cholesky.view(self.n_components, -1)[:, :: self.d_features + 1]), dim=1)
            log_prob = []
            for k in range(0, self.n_components):
                y = torch.matmul(X, self.precisions_cholesky[k]) - torch.matmul(self.means[k : k + 1], self.precisions_cholesky[k])
                log_prob.append(torch.sum(torch.square(y), dim=1))
            log_prob = torch.stack(log_prob, dim=1)
            log_prob = -0.5 * (self.d_features * torch.log(2 * self.pi) + log_prob) + log_det.view(1, self.n_components)

            log_weights = torch.log(self.weights)
            log_likelihoods = torch.logsumexp(log_prob + log_weights.view(1, self.n_components), dim=1)
        # print(log_likelihoods)
        return log_likelihoods

    @classmethod
    def np_to_param(cls, arr):
        return nn.Parameter(torch.from_numpy(arr).type(cls.data_type), requires_grad=True)

    @staticmethod
    def verify():
        import time
        np.random.seed(int(time.time()))
        X = np.random.rand(100, 4)
        gmm_sklearn = sklearn.mixture.GaussianMixture(n_components=5, max_iter=200, n_init=3, init_params='kmeans')
        gmm_sklearn.fit(X)
        gmm_pt = GMMDensityEstimator(gmm_sklearn).cuda()
        # check correctness
        X2 = np.random.rand(107, 4)
        ll_sklearn = gmm_sklearn.score_samples(X2)
        ll_pt = gmm_pt(torch.from_numpy(X2).type(gmm_pt.data_type).cuda())
        diff = np.absolute(ll_sklearn - ll_pt.detach().cpu().numpy())
        print(ll_sklearn[: 5], ll_pt[: 5], diff.mean())
        # check gradient connectivity
        embedding = nn.Linear(1, 4).cuda()
        opt = torch.optim.SGD(list(embedding.parameters()) + list(gmm_pt.parameters()), lr=1e-3)
        for _ in range(0, 5):
            print(embedding.weight.data[0, 0], gmm_pt.means[0, 0], gmm_pt.weights[0])
            opt.zero_grad()
            x = torch.randn(7, 1).cuda()
            x = gmm_pt(embedding(x))
            x.abs().sum().backward()
            opt.step()


current_image = {'batch': None}
def helper_show(boxes, log_likelihoods, title):
    import sklearn.utils
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    N = len(boxes)
    im_batch = current_image['batch']
    assert N == len(im_batch) == len(log_likelihoods)
    print(N, 'images')
    fig, axes = plt.subplots(3, N)
    for i in range(0, N):
        im_i = im_batch[i].transpose(1, 2, 0)[:, :, ::-1]
        if boxes[i] is None:
            axes[i][0].imshow(im_i)
            continue
        boxes_i, ll_i = boxes[i].detach().cpu().numpy(), log_likelihoods[i].detach().cpu().numpy()
        for j in range(0, 3):
            axes[i][j].imshow(im_i)
            axes[i][j].set_title('%s %d-%d' % (title, i, j))
            boxes_j, ll_j = sklearn.utils.shuffle(boxes_i, ll_i)
            boxes_j, ll_j = boxes_j[:16], ll_j[:16]
            for k in range(0, boxes_j.shape[0]):
                x1, y1, x2, y2 = boxes_j[k]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none')
                axes[i][j].add_patch(rect)
                axes[i][j].text((x1 + x2) / 2, (y1 + y2) / 2, '%.2f' % ll_j[k], size=12, color='red')
    plt.show()



if __name__ == '__main__':
    GMMDensityEstimator.verify()

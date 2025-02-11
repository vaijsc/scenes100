import argparse
import os
import sys
import gc
import json
import tqdm
import copy
import contextlib

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T_tv
from detectron2.structures import BoxMode

groundingdino_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GroundingDINO')
sys.path.append(groundingdino_dir)
import groundingdino.datasets.transforms as T_dino
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation import evaluate_masked

from inference_groundingdino import load_model, get_text_dict, BoxDataset
from dino.matcher import HungarianMatcher
from dino.criterion import SetCriterion

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # ERROR: huggingface/tokenizers: The current process just got forked, after parallelism has already been used.


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
bbox_rgbs = ['#FF0000', '#0000FF']


class PromptEnhancerDecoder(torch.nn.Module):
    def __init__(self):
        super(PromptEnhancerDecoder, self).__init__()
        self.decoder = torch.nn.TransformerDecoderLayer(256, 8, batch_first=True)

    def forward(self, text_embed, text_mask, im_embed1, im_mask1, im_embed2, im_mask2, sid):
        return self.decoder(
            text_embed,
            torch.cat([im_embed1, im_embed2], dim=1),
            tgt_key_padding_mask=text_mask,
            memory_key_padding_mask=torch.cat([im_mask1, im_mask2], dim=1)
        )


class PromptEnhancerDecoderSID(torch.nn.Module):
    def __init__(self):
        super(PromptEnhancerDecoderSID, self).__init__()
        self.decoder_dict = torch.nn.ModuleDict({
            video_id: torch.nn.TransformerDecoderLayer(256, 8, batch_first=True)
            for video_id in video_id_list
        })

    def forward(self, text_embed, text_mask, im_embed1, im_mask1, im_embed2, im_mask2, sid):
        assert text_embed.size(0) == len(sid)
        im_embed12 = torch.cat([im_embed1, im_embed2], dim=1)
        im_mask12 = torch.cat([im_mask1, im_mask2], dim=1)
        return torch.cat([
            self.decoder_dict[sid[_i]](
                text_embed[_i : _i + 1],
                im_embed12[_i : _i + 1],
                tgt_key_padding_mask=text_mask[_i : _i + 1],
                memory_key_padding_mask=im_mask12[_i : _i + 1]
            )
            for _i in range(0, len(sid))
        ], dim=0)


def get_grounding_output_no_classify(model, enhancer, text_dict_list, im_torch, sid, alpha):
    # breakpoint()
    assert alpha > -1e-5 and alpha < 1 + 1e-5
    with torch.no_grad():
        im_torch = im_torch.unsqueeze(0)
        if isinstance(im_torch, (list, torch.Tensor)):
            im_torch = nested_tensor_from_tensor_list(im_torch)
        model.features, model.poss = model.backbone(im_torch)

        srcs, masks = [], []
        for l, feat in enumerate(model.features):
            src, mask = feat.decompose()
            srcs.append(model.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # breakpoint()
        if model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, model.num_feature_levels):
                if l == _len_srcs:
                    src = model.input_proj[l](model.features[-1].tensors)
                else:
                    src = model.input_proj[l](srcs[-1])
                m = im_torch.mask
                mask = torch.nn.functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                model.poss.append(pos_l)
        # breakpoint()

    boxes_per_class, logits_per_class = [], []
    for text_dict in text_dict_list:
        # 'encoded_text':             bs, L, d_model
        # 'text_token_mask':          bs, L          True for nomask, False for mask
        # 'position_ids':             bs, L
        # 'text_self_attention_masks' bs, L, L       True for nomask, False for mask
        # breakpoint()
        srcs_prompt = [srcs[-2] + model.poss[-2], srcs[-1] + model.poss[-1]] # https://github.com/IDEA-Research/GroundingDINO/blob/df5b48a3efbaa64288d8d0ad09b748ac86f22671/groundingdino/models/GroundingDINO/transformer_vanilla.py#L72
        masks_prompt = masks[-2 :]
        srcs_prompt = [_s.view(_s.size(0), _s.size(1), -1).transpose(1, 2).detach() for _s in srcs_prompt] # B x H*W x D
        masks_prompt = [_m.view(_m.size(0), -1).detach() for _m in masks_prompt] # B x H*W, True for mask, False for nomask
        # breakpoint()
        encoded_text_enhanced = enhancer(text_dict['encoded_text'].detach(), torch.logical_not(text_dict['text_token_mask']), srcs_prompt[0], masks_prompt[0], srcs_prompt[1], masks_prompt[1], [sid])
        text_dict['encoded_text'] = text_dict['encoded_text'] * (1 - alpha) + encoded_text_enhanced * alpha

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = model.transformer(srcs, masks, input_query_bbox, model.poss, input_query_label, attn_mask, text_dict)

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], model.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack([
            layer_cls_embed(layer_hs, text_dict)
            for layer_cls_embed, layer_hs in zip(model.class_embed, hs)
        ])

        logits = outputs_class[-1][0] # (nq, 256)
        boxes = outputs_coord_list[-1][0] # (nq, 4)
        assert logits.size(0) == boxes.size(0) == 900, str(logits.size())
        logits = logits.max(dim=1)[0]
        logits, topk_idx = torch.topk(logits, 300)
        boxes = boxes[topk_idx]
        boxes_per_class.append(boxes)
        logits_per_class.append(logits)
        # breakpoint()
    return boxes_per_class, logits_per_class

def get_grounding_output_no_classify_prompt(model, enhancer, text_dict_list, im_torch, sid, alpha, enhancer_arch):
    # breakpoint()
    assert alpha > -1e-5 and alpha < 1 + 1e-5
    with torch.no_grad():
        im_torch = im_torch.unsqueeze(0)
        if isinstance(im_torch, (list, torch.Tensor)):
            im_torch = nested_tensor_from_tensor_list(im_torch)
        model.features, model.poss = model.backbone(im_torch)

        srcs, masks = [], []
        for l, feat in enumerate(model.features):
            src, mask = feat.decompose()
            srcs.append(model.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # breakpoint()
        if model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, model.num_feature_levels):
                if l == _len_srcs:
                    src = model.input_proj[l](model.features[-1].tensors)
                else:
                    src = model.input_proj[l](srcs[-1])
                m = im_torch.mask
                mask = torch.nn.functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                model.poss.append(pos_l)
        # breakpoint()

    boxes_per_class, logits_per_class = [], []
    for i, text_dict in enumerate(text_dict_list):
        # 'encoded_text':             bs, L, d_model
        # 'text_token_mask':          bs, L          True for nomask, False for mask
        # 'position_ids':             bs, L
        # 'text_self_attention_masks' bs, L, L       True for nomask, False for mask
        # breakpoint()
        # srcs_prompt = [srcs[-2] + model.poss[-2], srcs[-1] + model.poss[-1]] # https://github.com/IDEA-Research/GroundingDINO/blob/df5b48a3efbaa64288d8d0ad09b748ac86f22671/groundingdino/models/GroundingDINO/transformer_vanilla.py#L72
        # masks_prompt = masks[-2 :]
        # srcs_prompt = [_s.view(_s.size(0), _s.size(1), -1).transpose(1, 2).detach() for _s in srcs_prompt] # B x H*W x D
        # masks_prompt = [_m.view(_m.size(0), -1).detach() for _m in masks_prompt] # B x H*W, True for mask, False for nomask
        # breakpoint()
        if enhancer_arch == 'generic':
            encoded_text_enhanced = enhancer[i]
        else:
            encoded_text_enhanced = enhancer[i][sid]
        text_dict['encoded_text'] = text_dict['encoded_text'] * (1 - alpha) + encoded_text_enhanced * alpha

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = model.transformer(srcs, masks, input_query_bbox, model.poss, input_query_label, attn_mask, text_dict)

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], model.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack([
            layer_cls_embed(layer_hs, text_dict)
            for layer_cls_embed, layer_hs in zip(model.class_embed, hs)
        ])

        logits = outputs_class[-1][0] # (nq, 256)
        boxes = outputs_coord_list[-1][0] # (nq, 4)
        assert logits.size(0) == boxes.size(0) == 900, str(logits.size())
        logits = logits.max(dim=1)[0]
        logits, topk_idx = torch.topk(logits, 300)
        boxes = boxes[topk_idx]
        boxes_per_class.append(boxes)
        logits_per_class.append(logits)
        # breakpoint()
    return boxes_per_class, logits_per_class

class ColorJitterWrapper(object):
    def __init__(self):
        self.tf = T_tv.ColorJitter(brightness=0.1, hue=0.1)
    def __call__(self, image, target=None):
        return self.tf(image), target
class RandomAdjustSharpnessWrapper(object):
    def __init__(self):
        self.tf = T_tv.RandomAdjustSharpness(0.8, p=0.25)
    def __call__(self, image, target=None):
        return self.tf(image), target


def train_scenes100(args):
    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        RandomAdjustSharpnessWrapper(),
        ColorJitterWrapper(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.bert.eval()
    model.feat_map.eval()
    model.input_proj.eval()
    model.backbone.eval()
    model.transformer.train()
    model.bbox_embed.train()
    model.class_embed.train()
    for p in model.parameters():
        p.requires_grad = False

    loss_fn = SetCriterion(
        matcher     = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, focal_alpha=0.25),
        focal_alpha = 0.25,
        focal_gamma = 2,
        losses      = ['labels', 'boxes']
    )
    loss_history_dict = {
        'loss_ce': [],
        'loss_bbox': [],
        'loss_giou': [],
        'weighted': [],
    }

    with torch.no_grad():
        text_dict_list = get_text_dict(model, args.prompts)
    del model.bert # save some VRAM
    gc.collect()
    torch.cuda.empty_cache()

    if args.enhancer_arch == 'generic':
        enhancer = PromptEnhancerDecoder().cuda()
    elif args.enhancer_arch == 'sid':
        enhancer = PromptEnhancerDecoderSID().cuda()
    else:
        raise NotImplementedError
    if args.ckpt is not None:
        print('loading from', args.ckpt)
        enhancer.load_state_dict(torch.load(args.ckpt))
    enhancer.train()
    optimizer = torch.optim.Adam(enhancer.parameters(), lr=args.lr)

    dst_train = []
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        for im in images:
            im['video_id'] = video_id
            im['file_name'] = os.path.join(inputdir, 'masked', im['file_name'])
        images = sorted(images, key=lambda x: x['file_name'])
        dst_train.extend(copy.deepcopy(images[: args.shot]))
    print('%d-shot training set: %d images' % (args.shot, len(dst_train)))
    prefix = os.path.join(args.savedir, 'scenes100_%dshot_gdino_%s_enhancer_%s' % (args.shot, args.config, args.enhancer_arch))

    dst_train = BoxDataset(dst_train, image_transform)
    loader_train = torch.utils.data.DataLoader(dst_train, batch_size=args.image_batch_size, collate_fn=BoxDataset.collate, shuffle=True, num_workers=args.num_workers)
    iter_train = iter(loader_train)
    print(prefix)

    eval_iters = list(range(0, args.iters, args.eval_interval)) + [args.iters - 1]
    for it in tqdm.tqdm(range(0, args.iters), ascii=True):
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(loader_train)
            batch = next(iter_train)
        im_torch_batch, targets_per_class_batch, sid_batch = batch
        optimizer.zero_grad()

        if it <= args.iters / 3:
            alpha = 0.5
        elif it <= args.iters * 2 / 3:
            alpha = 0.75
        else:
            alpha = 1.0
        targets_all, outputs_all = [], {'pred_boxes': [], 'pred_logits': []}
        for im_torch, targets_per_class, sid in zip(im_torch_batch, targets_per_class_batch, sid_batch):
            boxes_per_class, logits_per_class = get_grounding_output_no_classify(model, enhancer, copy.deepcopy(text_dict_list), im_torch.cuda(), sid, alpha)
            for c, (_boxes, _logits, _t) in enumerate(zip(boxes_per_class, logits_per_class, targets_per_class)):
                _t['boxes'], _t['labels'] = _t['boxes'].cuda(), _t['labels'].cuda()
                targets_all.append(_t)
                outputs_all['pred_boxes'].append(_boxes)
                outputs_all['pred_logits'].append(_logits)
        outputs_all['pred_boxes'] = torch.stack(outputs_all['pred_boxes'], dim=0)
        outputs_all['pred_logits'] = torch.stack(outputs_all['pred_logits'], dim=0).unsqueeze(2)
        # print([_t['labels'].size() for _t in targets_all])
        # print(outputs_all['pred_boxes'].size(), outputs_all['pred_boxes'].min(), outputs_all['pred_boxes'].max())
        # print(outputs_all['pred_logits'].size(), outputs_all['pred_logits'].min(), outputs_all['pred_logits'].max())

        loss_dict = loss_fn(outputs_all, targets_all)
        L = loss_dict['loss_ce'] * args.cls_loss_coef + loss_dict['loss_bbox'] * args.bbox_loss_coef + loss_dict['loss_giou'] * args.giou_loss_coef
        L.backward()
        optimizer.step()
        # with torch.no_grad(): print(enhancer.decoder1.linear1.weight[:3, :3])

        loss_history_dict['loss_ce'].append(float(loss_dict['loss_ce'].item()))
        loss_history_dict['loss_bbox'].append(float(loss_dict['loss_bbox'].item()))
        loss_history_dict['loss_giou'].append(float(loss_dict['loss_giou'].item()))
        loss_history_dict['weighted'].append(float(L.item()))

        if it in eval_iters:
            torch.save(enhancer.state_dict(), prefix + '.pth')
            torch.cuda.empty_cache()
            if len(loss_history_dict['weighted']) > 200:
                plt.figure(figsize=(20, 20))
                loss_key_list = ['loss_ce', 'loss_bbox', 'loss_giou', 'weighted']
                loss_tensor = torch.tensor([loss_history_dict[k] for k in loss_key_list]).float()
                kernel = max(min(200, len(loss_history_dict['weighted']) // 100), 20)
                loss_smooth = torch.nn.functional.avg_pool1d(loss_tensor.unsqueeze(1), kernel)
                loss_smooth = loss_smooth[:, 0, :].detach().numpy()
                for i, loss_key in enumerate(loss_key_list):
                    plt.subplot(2, 2, i + 1)
                    plt.plot(np.arange(0, loss_smooth.shape[1]) * kernel, loss_smooth[i], 'r-')
                    plt.grid(True)
                    plt.xlim(0, args.iters)
                    plt.xlabel('Training Iterations')
                    plt.title(loss_key)
                plt.tight_layout()
                plt.savefig(prefix + '.pdf')
                plt.close()


def train_scenes100_prompt(args):
    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        RandomAdjustSharpnessWrapper(),
        ColorJitterWrapper(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.bert.eval()
    model.feat_map.eval()
    model.input_proj.eval()
    model.backbone.eval()
    model.transformer.train()
    model.bbox_embed.train()
    model.class_embed.train()
    for p in model.parameters():
        p.requires_grad = False

    loss_fn = SetCriterion(
        matcher     = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, focal_alpha=0.25),
        focal_alpha = 0.25,
        focal_gamma = 2,
        losses      = ['labels', 'boxes']
    )
    loss_history_dict = {
        'loss_ce': [],
        'loss_bbox': [],
        'loss_giou': [],
        'weighted': [],
    }

    with torch.no_grad():
        text_dict_list = get_text_dict(model, args.prompts)
    del model.bert # save some VRAM
    gc.collect()
    torch.cuda.empty_cache()

    enhancer = []
    if args.enhancer_arch == 'generic':
        for text_dict in text_dict_list:
            enhancer.append(torch.nn.Parameter(torch.zeros_like(text_dict['encoded_text'], requires_grad=True)))
        enhancer = torch.nn.ParameterList(enhancer)
        enhancer.train()
        optimizer = torch.optim.Adam(enhancer.parameters(), lr=args.lr)
    elif args.enhancer_arch == 'sid':
        param_list = []
        for text_dict in text_dict_list:
            enhancer_i = {}
            for sid in video_id_list:
                enhancer_i[sid] = torch.nn.Parameter(torch.zeros_like(text_dict['encoded_text'], requires_grad=True))
            enhancer.append(enhancer_i)
            param_list.append({'params': list(enhancer_i.values())})
        optimizer = torch.optim.Adam(param_list, lr=args.lr)
    else:
        raise NotImplementedError
    
    if args.ckpt is not None:
        print('loading from', args.ckpt)
        enhancer = torch.load(args.ckpt)

    # breakpoint()
    dst_train = []
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        for im in images:
            im['video_id'] = video_id
            im['file_name'] = os.path.join(inputdir, 'masked', im['file_name'])
        images = sorted(images, key=lambda x: x['file_name'])
        dst_train.extend(copy.deepcopy(images[: args.shot]))
    print('%d-shot training set: %d images' % (args.shot, len(dst_train)))
    prefix = os.path.join(args.savedir, 'scenes100_%dshot_gdino_%s_enhancer_%s' % (args.shot, args.config, args.enhancer_arch))

    dst_train = BoxDataset(dst_train, image_transform)
    loader_train = torch.utils.data.DataLoader(dst_train, batch_size=args.image_batch_size, collate_fn=BoxDataset.collate, shuffle=True, num_workers=args.num_workers)
    iter_train = iter(loader_train)
    print(prefix)

    eval_iters = list(range(0, args.iters, args.eval_interval)) + [args.iters - 1]
    for it in tqdm.tqdm(range(0, args.iters), ascii=True):
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(loader_train)
            batch = next(iter_train)
        im_torch_batch, targets_per_class_batch, sid_batch = batch
        optimizer.zero_grad()
        alpha = 1.0
        # alpha = min((it + 1) / args.iters + 0.5, 1)
        # if it <= args.iters / 3:
        #     alpha = 0.5
        # elif it <= args.iters * 2 / 3:
        #     alpha = 0.75
        # else:
        #     alpha = 1.0
        targets_all, outputs_all = [], {'pred_boxes': [], 'pred_logits': []}
        for im_torch, targets_per_class, sid in zip(im_torch_batch, targets_per_class_batch, sid_batch):
            boxes_per_class, logits_per_class = get_grounding_output_no_classify_prompt(model, enhancer, copy.deepcopy(text_dict_list), im_torch.cuda(), sid, alpha, args.enhancer_arch)
            for c, (_boxes, _logits, _t) in enumerate(zip(boxes_per_class, logits_per_class, targets_per_class)):
                _t['boxes'], _t['labels'] = _t['boxes'].cuda(), _t['labels'].cuda()
                targets_all.append(_t)
                outputs_all['pred_boxes'].append(_boxes)
                outputs_all['pred_logits'].append(_logits)
        outputs_all['pred_boxes'] = torch.stack(outputs_all['pred_boxes'], dim=0)
        outputs_all['pred_logits'] = torch.stack(outputs_all['pred_logits'], dim=0).unsqueeze(2)
        # print([_t['labels'].size() for _t in targets_all])
        # print(outputs_all['pred_boxes'].size(), outputs_all['pred_boxes'].min(), outputs_all['pred_boxes'].max())
        # print(outputs_all['pred_logits'].size(), outputs_all['pred_logits'].min(), outputs_all['pred_logits'].max())

        loss_dict = loss_fn(outputs_all, targets_all)
        L = loss_dict['loss_ce'] * args.cls_loss_coef + loss_dict['loss_bbox'] * args.bbox_loss_coef + loss_dict['loss_giou'] * args.giou_loss_coef
        L.backward()
        optimizer.step()
        # with torch.no_grad(): print(enhancer.decoder1.linear1.weight[:3, :3])

        loss_history_dict['loss_ce'].append(float(loss_dict['loss_ce'].item()))
        loss_history_dict['loss_bbox'].append(float(loss_dict['loss_bbox'].item()))
        loss_history_dict['loss_giou'].append(float(loss_dict['loss_giou'].item()))
        loss_history_dict['weighted'].append(float(L.item()))

        if it in eval_iters:
            torch.save(enhancer, prefix + '_prompt.pth')
            torch.cuda.empty_cache()
            if len(loss_history_dict['weighted']) > 200:
                plt.figure(figsize=(20, 20))
                loss_key_list = ['loss_ce', 'loss_bbox', 'loss_giou', 'weighted']
                loss_tensor = torch.tensor([loss_history_dict[k] for k in loss_key_list]).float()
                kernel = max(min(200, len(loss_history_dict['weighted']) // 100), 20)
                loss_smooth = torch.nn.functional.avg_pool1d(loss_tensor.unsqueeze(1), kernel)
                loss_smooth = loss_smooth[:, 0, :].detach().numpy()
                for i, loss_key in enumerate(loss_key_list):
                    plt.subplot(2, 2, i + 1)
                    plt.plot(np.arange(0, loss_smooth.shape[1]) * kernel, loss_smooth[i], 'r-')
                    plt.grid(True)
                    plt.xlim(0, args.iters)
                    plt.xlabel('Training Iterations')
                    plt.title(loss_key)
                plt.tight_layout()
                plt.savefig(prefix + '_prompt.pdf')
                plt.close()


@torch.no_grad()
def eval_scenes100(args):
    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
    with torch.no_grad():
        text_dict_list = get_text_dict(model, args.prompts)
    if args.enhancer_arch == 'generic':
        enhancer = PromptEnhancerDecoder().cuda()
    elif args.enhancer_arch == 'sid':
        enhancer = PromptEnhancerDecoderSID().cuda()
    else:
        raise NotImplementedError
    if args.ckpt is not None:
        print('loading from', args.ckpt)
        enhancer.load_state_dict(torch.load(args.ckpt))
    enhancer.eval()

    images_all_video = []
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        for im in images:
            im['video_id'] = video_id
            im['file_name'] = os.path.join(inputdir, 'unmasked', im['file_name'])
        images_all_video.extend(images)
    loader = torch.utils.data.DataLoader(
        BoxDataset(images_all_video, image_transform, training=False),
        batch_size=None, shuffle=False, num_workers=4
    )

    detections = {}
    for im_torch, im in tqdm.tqdm(loader, ascii=True, total=len(images_all_video)):
        if not im['video_id'] in detections:
            detections[im['video_id']] = []
        im['file_name'] = os.path.basename(im['file_name'])
        im['annotations'] = []
        boxes_per_class, logits_per_class = get_grounding_output_no_classify(model, enhancer, copy.deepcopy(text_dict_list), im_torch.cuda(), im['video_id'], 1.0)
        scores_per_class = [_logits.sigmoid().detach().cpu().numpy() for _logits in logits_per_class]
        for c in range(0, len(boxes_per_class)):
            boxes_per_class[c][:, 0] *= im['width']
            boxes_per_class[c][:, 2] *= im['width']
            boxes_per_class[c][:, 1] *= im['height']
            boxes_per_class[c][:, 3] *= im['height']
            _xyxy = torch.stack([
                boxes_per_class[c][:, 0] - boxes_per_class[c][:, 2] / 2,
                boxes_per_class[c][:, 1] - boxes_per_class[c][:, 3] / 2,
                boxes_per_class[c][:, 0] + boxes_per_class[c][:, 2] / 2,
                boxes_per_class[c][:, 1] + boxes_per_class[c][:, 3] / 2,
            ], dim=1).detach().cpu().numpy()
            for box, s in zip(_xyxy, scores_per_class[c]):
                im['annotations'].append({'bbox': list(map(float, box)), 'segmentation': [], 'category_id': c, 'score': float(s), 'bbox_mode': BoxMode.XYXY_ABS})
        detections[im['video_id']].append(im)

    APs_all = {}
    for video_id in tqdm.tqdm(detections, ascii=True):
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs_all[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)
        del APs_all[video_id]['raw']
    categories = ['person', 'vehicle', 'overall', 'weighted']
    print('videos average:')
    for c in categories:
        _AP_videos = np.array([APs_all[v]['results'][c] for v in APs_all]) * 100
        print(c, _AP_videos[_AP_videos[:, 0] >= 0].mean(axis=0))


@torch.no_grad()
def eval_scenes100_prompt(args):
    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
    with torch.no_grad():
        text_dict_list = get_text_dict(model, args.prompts)
    # if args.enhancer_arch == 'generic':
    #     enhancer = PromptEnhancerDecoder().cuda()
    # elif args.enhancer_arch == 'sid':
    #     enhancer = PromptEnhancerDecoderSID().cuda()
    # else:
    #     raise NotImplementedError
    assert args.ckpt is not None, "checkpoint is None!"
    if args.ckpt is not None:
        print('loading from', args.ckpt)
        enhancer = torch.load(args.ckpt)
    # enhancer.eval()

    images_all_video = []
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        for im in images:
            im['video_id'] = video_id
            im['file_name'] = os.path.join(inputdir, 'unmasked', im['file_name'])
        images_all_video.extend(images)
    loader = torch.utils.data.DataLoader(
        BoxDataset(images_all_video, image_transform, training=False),
        batch_size=None, shuffle=False, num_workers=4
    )

    detections = {}
    for im_torch, im in tqdm.tqdm(loader, ascii=True, total=len(images_all_video)):
        if not im['video_id'] in detections:
            detections[im['video_id']] = []
        im['file_name'] = os.path.basename(im['file_name'])
        im['annotations'] = []
        boxes_per_class, logits_per_class = get_grounding_output_no_classify_prompt(model, enhancer, copy.deepcopy(text_dict_list), im_torch.cuda(), im['video_id'], 1.0, args.enhancer_arch)
        scores_per_class = [_logits.sigmoid().detach().cpu().numpy() for _logits in logits_per_class]
        for c in range(0, len(boxes_per_class)):
            boxes_per_class[c][:, 0] *= im['width']
            boxes_per_class[c][:, 2] *= im['width']
            boxes_per_class[c][:, 1] *= im['height']
            boxes_per_class[c][:, 3] *= im['height']
            _xyxy = torch.stack([
                boxes_per_class[c][:, 0] - boxes_per_class[c][:, 2] / 2,
                boxes_per_class[c][:, 1] - boxes_per_class[c][:, 3] / 2,
                boxes_per_class[c][:, 0] + boxes_per_class[c][:, 2] / 2,
                boxes_per_class[c][:, 1] + boxes_per_class[c][:, 3] / 2,
            ], dim=1).detach().cpu().numpy()
            for box, s in zip(_xyxy, scores_per_class[c]):
                im['annotations'].append({'bbox': list(map(float, box)), 'segmentation': [], 'category_id': c, 'score': float(s), 'bbox_mode': BoxMode.XYXY_ABS})
        detections[im['video_id']].append(im)

    APs_all = {}
    for video_id in tqdm.tqdm(detections, ascii=True):
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs_all[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)
        del APs_all[video_id]['raw']
    categories = ['person', 'vehicle', 'overall', 'weighted']
    print('videos average:')
    for c in categories:
        _AP_videos = np.array([APs_all[v]['results'][c] for v in APs_all]) * 100
        print(c, _AP_videos[_AP_videos[:, 0] >= 0].mean(axis=0))

def inspect_enhancer(args):
    model = load_model(args.config).cuda()
    model.eval()
    with torch.no_grad():
        text_dict_list = get_text_dict(model, args.prompts)
    enhancer_sid = PromptEnhancerSID().cuda()
    enhancer_sid.load_state_dict(torch.load(args.ckpt.split('*')[0]))
    enhancer_self = PromptEnhancerSA().cuda()
    enhancer_self.load_state_dict(torch.load(args.ckpt.split('*')[1]))

    embeddings = [[], []]
    for c, text_dict in enumerate(text_dict_list):
        print(c, text_dict['tokenized_input_ids'][0])
        embeddings[c].append(text_dict['encoded_text'][0])
        embeddings[c].append(enhancer_self(text_dict['encoded_text'].detach(), torch.logical_not(text_dict['text_token_mask']))[0])
        for sid in video_id_list:
            embeddings[c].append(enhancer_sid(text_dict['encoded_text'].detach(), torch.logical_not(text_dict['text_token_mask']), [sid])[0])
        embeddings[c] = torch.stack(embeddings[c], dim=0).detach().cpu().numpy() # 102 x 4 x 256, 102 x 10 x 256

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, n_jobs=20, init='pca', learning_rate='auto')
    embeddings_tsne = tsne.fit_transform(np.concatenate([embeddings[0].reshape(408, 256), embeddings[1].reshape(1020, 256)], axis=0))
    embeddings_tsne -= embeddings_tsne[0]
    vec = embeddings_tsne[1] - embeddings_tsne[0]
    if vec[0] < 0:
        embeddings_tsne *= [-1, 1]
    phi = np.arctan(vec[1] / vec[0])
    R = np.array([
        [ np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)]
    ])
    embeddings_tsne = np.matmul(embeddings_tsne, R.T)
    embeddings_tsne[:, 0] -= embeddings_tsne[:, 0].min()
    embeddings_tsne[:, 1] -= embeddings_tsne[:, 1].min()
    embeddings_tsne /= embeddings_tsne.max()

    embeddings_tsne = [embeddings_tsne[: 408].reshape(102, 4, 2), embeddings_tsne[408 :].reshape(102, 10, 2)]
    plt.figure(figsize=(12, 6))

    words = list(filter(lambda x: len(x) > 1, text_dict_list[0]['caption'].split(' ')))
    assert len(words) == 1
    print(words)
    plt.subplot(2, 4, 1)
    _xy = embeddings_tsne[0][0 : 1, 1, :]
    plt.scatter(_xy[:, 0], _xy[:, 1], s=64, c='k', alpha=0.7, marker='o')
    _xy = embeddings_tsne[0][1 : 2, 1, :]
    plt.scatter(_xy[:, 0], _xy[:, 1], s=64, c='r', alpha=0.7, marker='o')
    _xy = embeddings_tsne[0][2 :  , 1, :]
    plt.scatter(_xy[:, 0], _xy[:, 1], s=4,  c='b', alpha=0.5, marker='.')
    plt.legend(['original', 'self-attention', 'SID-adaptive'])
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.title('embedding of \"%s\"' % words[0])
    plt.grid(True)

    word_idx = [1, 3, 5, 7]
    words = list(filter(lambda x: len(x) > 1, text_dict_list[1]['caption'].split(' ')))
    assert len(words) == len(word_idx)
    print(words)
    for i in range(0, len(word_idx)):
        plt.subplot(2, 4, 5 + i)
        _xy = embeddings_tsne[1][0 : 1, i, :]
        plt.scatter(_xy[:, 0], _xy[:, 1], s=64, c='k', alpha=0.7, marker='o')
        _xy = embeddings_tsne[1][1 : 2, i, :]
        plt.scatter(_xy[:, 0], _xy[:, 1], s=64, c='r', alpha=0.7, marker='o')
        _xy = embeddings_tsne[1][2 :  , i, :]
        plt.scatter(_xy[:, 0], _xy[:, 1], s=4,  c='b', alpha=0.5, marker='.')
        plt.legend(['original', 'self-attention', 'SID-adaptive'])
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.title('embedding of \"%s\"' % words[i])
        plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('gdino_bert.pdf')


def reconstruct_input(args):
    import transformers
    def _bert_forward_embeddings(model, inputs_embeds, type_embeds, pos_embeds, attention_mask, input_ids):
        token_embed = inputs_embeds + type_embeds + pos_embeds
        token_embed = model.bert.embeddings.LayerNorm(token_embed)
        token_embed = model.bert.embeddings.dropout(token_embed)
        encoder_outputs = model.bert.encoder(
            token_embed,
            attention_mask=model.bert.get_extended_attention_mask(attention_mask, input_ids.size())
        )
        features = transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state = encoder_outputs[0],
            pooler_output     = model.bert.pooler(encoder_outputs[0]),
            past_key_values   = encoder_outputs.past_key_values,
            hidden_states     = encoder_outputs.hidden_states,
            attentions        = encoder_outputs.attentions,
            cross_attentions  = encoder_outputs.cross_attentions,
        )
        return model.feat_map(features['last_hidden_state'])

    # must deepcopy all arguments before calling
    def _train_input(model, token_embed_list, text_dict_list, encoded_text_list_enhanced, prefix):
        print(prefix)
        with torch.no_grad():
            for c in range(0, 2):
                print(text_dict_list[c]['caption'], text_dict_list[c]['tokenized_input_ids'][0].detach().cpu().numpy())
                assert text_dict_list[c]['tokenized_input_ids'].size(0) == 1
                assert text_dict_list[c]['tokenized_input_ids'][0, 0] == 101
                assert text_dict_list[c]['tokenized_input_ids'][0,-1] == 102
                assert torch.isclose(text_dict_list[c]['encoded_text'], _bert_forward_embeddings(model, token_embed_list[c]['inputs_embeds'], token_embed_list[c]['type_embeds'], token_embed_list[c]['pos_embeds'], text_dict_list[c]['text_self_attention_masks'], text_dict_list[c]['tokenized_input_ids'])).all()
        # model.bert.train()
        # model.bert.eval()

        inputs_embeds_params = [copy.deepcopy(token_embed['inputs_embeds']) for token_embed in token_embed_list]
        with torch.no_grad():
            _mean, _std = model.bert.embeddings.word_embeddings.weight.data.mean(), model.bert.embeddings.word_embeddings.weight.data.std()
            inputs_embeds_params = [torch.randn(p.size()).to(p.device) * _std + _mean for p in inputs_embeds_params]
        inputs_embeds_params = [torch.nn.parameter.Parameter(p, requires_grad=True) for p in inputs_embeds_params]
        optimizer = torch.optim.Adam(inputs_embeds_params, lr=args.lr)
        eval_iters = list(range(0, args.iters, args.eval_interval)) + [args.iters - 1]
        loss_history = []
        for it in tqdm.tqdm(range(0, args.iters), ascii=True):
            if it in eval_iters:
                with torch.no_grad():
                    fp = open(prefix + '.txt', 'a')
                    fp.write('iter %d\n' % it)
                    _all_embeds = model.bert.embeddings.word_embeddings.weight.data
                    for c in range(0, len(inputs_embeds_params)):
                        _dists = inputs_embeds_params[c][0, 1 : -1].unsqueeze(1) - _all_embeds.unsqueeze(0)
                        _dists = _dists.square().sum(-1)
                        _, _topk = torch.topk(_dists, 5, dim=1, largest=False)
                        # _dists = (inputs_embeds_params[c][0, 1 : -1].unsqueeze(1) * _all_embeds.unsqueeze(0)).sum(-1)
                        # _, _topk = torch.topk(_dists, 5, dim=1)
                        # _dists = (torch.nn.functional.normalize(inputs_embeds_params[c][0, 1 : -1], dim=1).unsqueeze(1) * torch.nn.functional.normalize(_all_embeds, dim=1).unsqueeze(0)).sum(-1)
                        # _, _topk = torch.topk(_dists, 5, dim=1)
                        fp.write('class-%s\n' % c)
                        for _idx in _topk:
                            fp.write(str(model.tokenizer.convert_ids_to_tokens(_idx)) + '\n')
                    fp.close()
                if len(loss_history) > 10:
                    plt.figure(figsize=(8, 8))
                    plt.plot(np.arange(0, len(loss_history)), [l[0] for l in loss_history], 'r-')
                    plt.plot(np.arange(0, len(loss_history)), [l[1] for l in loss_history], 'b-')
                    plt.legend(['class-0 loss', 'class-1 loss'])
                    plt.xlim(0, args.iters)
                    plt.grid(True)
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(prefix + '.pdf')

            optimizer.zero_grad()
            loss_history.append([])
            for params, token_embed, text_dict, encoded_text_enhanced in zip(inputs_embeds_params, token_embed_list, text_dict_list, encoded_text_list_enhanced):
                out = _bert_forward_embeddings(
                    model,
                    params,
                    token_embed['type_embeds'].detach(),
                    token_embed['pos_embeds'].detach(),
                    text_dict['text_self_attention_masks'].detach(),
                    text_dict['tokenized_input_ids'].detach(),
                )
                L = (out - encoded_text_enhanced).square().mean().sqrt()
                L.backward()
                params.grad[:, 0, :] *= 0.0 # mask-out <SOS> and <EOS>
                params.grad[:,-1, :] *= 0.0
                loss_history[-1].append(float(L.item()))
            optimizer.step()

    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        RandomAdjustSharpnessWrapper(),
        ColorJitterWrapper(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = load_model(args.config).cuda()
    model.bert.eval()
    model.feat_map.eval()
    model.input_proj.eval()
    model.backbone.eval()
    model.transformer.eval()
    model.bbox_embed.eval()
    model.class_embed.eval()
    if args.enhancer_arch == 'generic':
        enhancer = PromptEnhancerDecoder().cuda()
    elif args.enhancer_arch == 'sid':
        enhancer = PromptEnhancerDecoderSID().cuda()
    else:
        raise NotImplementedError
    print('loading from', args.ckpt)
    enhancer.load_state_dict(torch.load(args.ckpt))
    enhancer.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in enhancer.parameters():
        p.requires_grad = False

    # original BERT embeddings & features
    with torch.no_grad():
        text_dict_list = get_text_dict(model, args.prompts)
        token_embed_list = []
        for text_dict in text_dict_list:
            position_embeds = model.bert.embeddings.position_embeddings(text_dict['position_ids'])
            token_embed_list.append({
                'inputs_embeds': model.bert.embeddings.word_embeddings(text_dict['tokenized_input_ids']),
                'type_embeds': model.bert.embeddings.token_type_embeddings(text_dict['tokenized_type_ids']),
                'pos_embeds': position_embeds if model.bert.embeddings.position_embedding_type == 'absolute' else torch.zeros_like(position_embeds)
            })
        for c in range(0, 2):
            print('caption:    ', text_dict_list[c]['caption'])
            print('tokens:     ', text_dict_list[c]['tokenized_input_ids'].detach().cpu().numpy())
            print('positions:  ', text_dict_list[c]['position_ids'].detach().cpu().numpy())
            print('types:      ', text_dict_list[c]['tokenized_type_ids'].detach().cpu().numpy())
            print('token embed:', token_embed_list[c]['inputs_embeds'].size())
            print('type embed: ', token_embed_list[c]['type_embeds'].size())
            print('pos embed:  ', token_embed_list[c]['pos_embeds'].size())
            print('features:   ', text_dict_list[c]['encoded_text'].size())
            # check matching
            print('check:      ', (text_dict_list[c]['encoded_text'] - _bert_forward_embeddings(model, token_embed_list[c]['inputs_embeds'], token_embed_list[c]['type_embeds'], token_embed_list[c]['pos_embeds'], text_dict_list[c]['text_self_attention_masks'], text_dict_list[c]['tokenized_input_ids'])).abs().max())

    images_1shot = []
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            im = sorted(json.load(fp), key=lambda x: x['file_name'])[0]
        im['video_id'] = video_id
        im['file_name'] = os.path.join(inputdir, 'masked', im['file_name'])
        images_1shot.append(im)
    dst = BoxDataset(images_1shot, image_transform, training=False)

    for im_torch, im in dst:
        with torch.no_grad():
            im_torch = im_torch.unsqueeze(0).cuda()
            if isinstance(im_torch, (list, torch.Tensor)):
                im_torch = nested_tensor_from_tensor_list(im_torch)
            model.features, model.poss = model.backbone(im_torch)
            srcs, masks = [], []
            for l, feat in enumerate(model.features):
                src, mask = feat.decompose()
                srcs.append(model.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            if model.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, model.num_feature_levels):
                    if l == _len_srcs:
                        src = model.input_proj[l](model.features[-1].tensors)
                    else:
                        src = model.input_proj[l](srcs[-1])
                    m = im_torch.mask
                    mask = torch.nn.functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    model.poss.append(pos_l)

            encoded_text_list_enhanced = []
            for text_dict in text_dict_list:
                # 'encoded_text':             bs, L, d_model
                # 'text_token_mask':          bs, L          True for nomask, False for mask
                # 'position_ids':             bs, L
                # 'text_self_attention_masks' bs, L, L       True for nomask, False for mask

                srcs_prompt = [srcs[-2] + model.poss[-2], srcs[-1] + model.poss[-1]] # https://github.com/IDEA-Research/GroundingDINO/blob/df5b48a3efbaa64288d8d0ad09b748ac86f22671/groundingdino/models/GroundingDINO/transformer_vanilla.py#L72
                masks_prompt = masks[-2 :]
                srcs_prompt = [_s.view(_s.size(0), _s.size(1), -1).transpose(1, 2).detach() for _s in srcs_prompt] # B x H*W x D
                masks_prompt = [_m.view(_m.size(0), -1).detach() for _m in masks_prompt] # B x H*W, True for mask, False for nomask
                encoded_text_list_enhanced.append(enhancer(text_dict['encoded_text'], torch.logical_not(text_dict['text_token_mask']), srcs_prompt[0], masks_prompt[0], srcs_prompt[1], masks_prompt[1], [im['video_id']]))

        _train_input(
            model,
            copy.deepcopy(token_embed_list),
            copy.deepcopy(text_dict_list),
            copy.deepcopy(encoded_text_list_enhanced),
            args.ckpt + '_%s_reconstruct' % im['video_id']
        )
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Grounding DINO example', add_help=True)
    parser.add_argument('--opt', type=str)
    parser.add_argument('--config', '-c', type=str, default='swint', choices=['swint', 'swinb'], help='path to config file')
    parser.add_argument('--box_threshold', type=float, default=0.05, help='box threshold')
    parser.add_argument('--prompts', type=str, nargs='+', default=['person .', 'vehicle . car . bus . truck .'])

    parser.add_argument('--enhancer_arch', type=str, default='generic', choices=['sid', 'generic'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--cls_loss_coef', type=float, default=2.0)
    parser.add_argument('--bbox_loss_coef', type=float, default=5.0)
    parser.add_argument('--giou_loss_coef', type=float, default=2.0)

    parser.add_argument('--shot', type=int, default=1, help='few-shot learning')
    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=2, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--prompt_only', action='store_true')

    parser.add_argument('--savedir', type=str, default='.')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.opt == 'train':
        if args.prompt_only:
            train_scenes100_prompt(args)
        else:
            train_scenes100(args)
    if args.opt == 'eval':
        if args.prompt_only:
            eval_scenes100_prompt(args)
        else:
            eval_scenes100(args)
    if args.opt == 're':
        reconstruct_input(args)
    if args.opt == 'inspect':
        inspect_enhancer(args)

'''
python finetune_groundingdino.py --opt train --iters 1000 --eval_interval 201 --image_batch_size 2 --num_workers 2 --enhancer_arch generic --shot 1
python finetune_groundingdino.py --opt eval --enhancer_arch generic --ckpt scenes100_1shot_gdino_swint_enhancer_generic.pth

python finetune_groundingdino.py --opt train --iters 800 --eval_interval 201 --image_batch_size 2 --num_workers 2 --enhancer_arch sid --shot 1
python finetune_groundingdino.py --opt eval --enhancer_arch sid --ckpt scenes100_1shot_gdino_swint_enhancer_sid.pth

for A in generic sid ; do
  for S in 1 3 30 ; do
    python finetune_groundingdino.py --opt train --iters 4000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --enhancer_arch ${A} --shot ${S} --prompts "people ." "vehicle ."
  done
done

python finetune_groundingdino.py --opt inspect --ckpt /mnt/f/intersections_results/accv24/gdino_enhancer/scenes100_1shot_gdino_swint_enhancer_sid.pth*/mnt/f/intersections_results/accv24/gdino_enhancer/scenes100_1shot_gdino_swint_enhancer_self.pth

CUDA_VISIBLE_DEVICES=0 nohup python finetune_groundingdino.py --opt train --iters 4000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --enhancer_arch generic --shot 1 --prompts "vacuum ." "light bulb ." &> log.gdino.enhancer.generic.01.log &
CUDA_VISIBLE_DEVICES=1 nohup python finetune_groundingdino.py --opt train --iters 4000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --enhancer_arch generic --shot 3 --prompts "vacuum ." "light bulb ." &> log.gdino.enhancer.generic.03.log &
CUDA_VISIBLE_DEVICES=3 nohup python finetune_groundingdino.py --opt train --iters 4000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --enhancer_arch generic --shot 30 --prompts "vacuum ." "light bulb ." &> log.gdino.enhancer.generic.30.log &
CUDA_VISIBLE_DEVICES=4 nohup python finetune_groundingdino.py --opt train --iters 4000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --enhancer_arch sid --shot 1 --prompts "vacuum ." "light bulb ." &> log.gdino.enhancer.sid.01.log &
CUDA_VISIBLE_DEVICES=5 nohup python finetune_groundingdino.py --opt train --iters 4000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --enhancer_arch sid --shot 3 --prompts "vacuum ." "light bulb ." &> log.gdino.enhancer.sid.03.log &
CUDA_VISIBLE_DEVICES=6 nohup python finetune_groundingdino.py --opt train --iters 4000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --enhancer_arch sid --shot 30 --prompts "vacuum ." "light bulb ." &> log.gdino.enhancer.sid.30.log &

python finetune_groundingdino.py --opt re --iters 10000 --eval_interval 2001 --enhancer_arch generic --ckpt /mnt/f/intersections_results/accv24/gdino_enhancer/promptsD/scenes100_1shot_gdino_swint_enhancer_generic.pth --prompts "gens . volk ." "voiture . wagen ."
python finetune_groundingdino.py --opt re --iters 10000 --eval_interval 2001 --enhancer_arch generic --ckpt /mnt/f/intersections_results/accv24/gdino_enhancer/promptsB/scenes100_1shot_gdino_swint_enhancer_generic.pth --prompts "people ." "vehicle ."
python finetune_groundingdino.py --opt re --iters 10000 --eval_interval 2001 --enhancer_arch generic --ckpt /mnt/f/intersections_results/accv24/gdino_enhancer/promptsA/scenes100_1shot_gdino_swint_enhancer_generic.pth --prompts "person . people . pedestrian . driver ." "vehicle . car . bus . truck ."
'''

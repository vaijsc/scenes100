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
from evaluation import eval_AP

from inference_groundingdino_accv import load_model, get_text_dict, BoxDataset, get_scenes100_images, evaluate_scenes100_masked_fewshot, get_egoper_images, get_hoist_images, get_ovcoco_images, get_birdsai_images, get_rareplanes_images, insert_lora, insert_losa, insert_res_tuner, count_parameters, LowRankTransformerDecoderLayer
from dino.matcher import HungarianMatcher
from dino.criterion import SetCriterion

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # ERROR: huggingface/tokenizers: The current process just got forked, after parallelism has already been used.


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
bbox_rgbs = ['#FF0000', '#0000FF']


class PromptEnhancer(torch.nn.Module):
    def __init__(self, rank_qkv=16, rank_ff=64):
        super(PromptEnhancer, self).__init__()
        # self.decoder = torch.nn.TransformerDecoderLayer(256, 8, batch_first=True)
        self.decoder = LowRankTransformerDecoderLayer(d_model=256, n_heads=8, d_ff=rank_ff, rank=rank_qkv, batch_first=True)
    def forward(self, text_embed, text_mask, im_embed1, im_mask1, im_embed2, im_mask2):
        return self.decoder(
            text_embed,
            torch.cat([im_embed1, im_embed2], dim=1),
            tgt_key_padding_mask=text_mask,
            memory_key_padding_mask=torch.cat([im_mask1, im_mask2], dim=1)
        )


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


def model_forward_single_image(model, text_dict, im_torch, arch):
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
    if arch == 'enhancer':
        srcs_prompt = [srcs[-2] + model.poss[-2], srcs[-1] + model.poss[-1]] # https://github.com/IDEA-Research/GroundingDINO/blob/df5b48a3efbaa64288d8d0ad09b748ac86f22671/groundingdino/models/GroundingDINO/transformer_vanilla.py#L72
        masks_prompt = masks[-2 :]
        srcs_prompt = [_s.view(_s.size(0), _s.size(1), -1).transpose(1, 2).detach() for _s in srcs_prompt] # B x H*W x D
        masks_prompt = [_m.view(_m.size(0), -1).detach() for _m in masks_prompt] # B x H*W, True for mask, False for nomask
        text_dict['encoded_text'] = model.enhancer(text_dict['encoded_text'].detach(), torch.logical_not(text_dict['text_token_mask']), srcs_prompt[0], masks_prompt[0], srcs_prompt[1], masks_prompt[1])

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
    return boxes, logits

def train_few_shot(args):
    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        RandomAdjustSharpnessWrapper(),
        ColorJitterWrapper(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)

    if args.arch == 'lora':
        model = load_model(args.config).cuda()
        model.eval()
        num_model_params = count_parameters(model)
        for p in model.parameters():
            p.requires_grad = False
        insert_lora(model, args)

        if args.ckpt is not None:
            print('loading from', args.ckpt)
            lora_state_dict = torch.load(args.ckpt)
            # Load each saved LoRA layer's state dict back into the model
            for i, lora_layer in enumerate(model.lora_layers):
                lora_layer.load_state_dict(lora_state_dict[f"lora_{i}"])
        
        for layer in model.lora_layers:
            layer.train()

        lora_params = []
        for lora_layer in model.lora_layers:
            lora_params.extend(lora_layer.parameters())
        optimizer = torch.optim.Adam(lora_params, lr=args.lr)
        print(f"LoRA params take {count_parameters(model.lora_layers)/num_model_params*100:.2f}%")

    elif args.arch == 'losa':
        model = load_model(args.config).cuda()
        model.eval()
        num_model_params = count_parameters(model)
        for p in model.parameters():
            p.requires_grad = False
        insert_losa(model, args)

        if args.ckpt is not None:
            print('loading from', args.ckpt)
            losa_state_dict = torch.load(args.ckpt)
            model.transformer.encoder.losa.load_state_dict(losa_state_dict)

        model.transformer.encoder.losa.train()
        optimizer = torch.optim.Adam(model.transformer.encoder.losa.parameters(), lr=args.lr)
        print(f"LoSA params take {count_parameters(model.transformer.encoder.losa)/num_model_params*100:.2f}%")

    elif args.arch == 'res_tuner':
        model = load_model(args.config).cuda()
        model.eval()
        num_model_params = count_parameters(model)
        for p in model.parameters():
            p.requires_grad = False
        insert_res_tuner(model, args)

        if args.ckpt is not None:
            print('loading from', args.ckpt)
            res_tuner_state_dict = torch.load(args.ckpt)
            # Load each saved ResTuner layer's state dict back into the model
            for i, tuner_layer in enumerate(model.tuner_layers):
                tuner_layer.load_state_dict(res_tuner_state_dict[f"tuner_{i}"])
        
        for layer in model.tuner_layers:
            layer.train()

        tuner_params = []
        for tuner_layer in model.tuner_layers:
            tuner_params.extend(tuner_layer.parameters())
        optimizer = torch.optim.Adam(tuner_params, lr=args.lr)
        with torch.no_grad():
            text_dict_cache = get_text_dict(model, args.prompt)
        print(f"ResTuner params take {count_parameters(model.tuner_layers)/num_model_params*100:.2f}%")

    elif args.arch == 'bitfit':
        model = load_model(args.config).cuda()
        model.eval()
        num_model_params = count_parameters(model)
        for p in model.parameters():
            p.requires_grad = False
        
        biases = []
        param_count = 0
        for name, param in model.named_parameters():
            if 'bias' in name and 'bert' in name:
                biases.append(param)
                param.requires_grad = True
                param_count += param.numel()

        if args.ckpt is not None:
            print('loading from', args.ckpt)
            bias_dict = torch.load(args.ckpt)
            # Load each bias term back into the model
            with torch.no_grad():  # Prevents tracking in the computation graph
                for name, param in model.named_parameters():
                    if name in bias_dict:
                        param.copy_(bias_dict[name])

        optimizer = torch.optim.Adam(biases, lr=args.lr)
        print(f"BitFit params take {param_count/num_model_params*100:.2f}%")

    elif args.arch == 'prompt':
        model = load_model(args.config).cuda()
        model.eval()
        num_model_params = count_parameters(model)
        for p in model.parameters():
            p.requires_grad = False
              
        with torch.no_grad():
            text_dict_cache = get_text_dict(model, args.prompt)
        del model.bert     

        for key in text_dict_cache:
            if key != 'caption':
                text_dict_cache[key] = text_dict_cache[key].detach()
        # Zero-init
        # text_dict_cache['encoded_text'] = torch.zeros_like(text_dict_cache['encoded_text'], requires_grad=True)
        
        # Init from encoded text
        text_dict_cache['encoded_text'].requires_grad = True

        if args.ckpt is not None:
            print('loading from', args.ckpt)
            text_dict_cache['encoded_text'] = torch.load(args.ckpt)

        optimizer = torch.optim.Adam([text_dict_cache['encoded_text']], lr=args.lr)
        print(f"Prompt params take {text_dict_cache['encoded_text'].numel()/num_model_params*100:.2f}%")

    elif args.arch == 'enhancer':
        model = load_model(args.config).cuda()
        model.eval()
        num_model_params = count_parameters(model)
        for p in model.parameters():
            p.requires_grad = False

        model.enhancer = PromptEnhancer(args.enhancer_r_qkv, args.enhancer_r_ff).cuda()
        if args.ckpt is not None:
            print('loading from', args.ckpt)
            model.enhancer.load_state_dict(torch.load(args.ckpt))
        model.enhancer.train()
        optimizer = torch.optim.Adam(model.enhancer.parameters(), lr=args.lr)

        with torch.no_grad():
            text_dict_cache = get_text_dict(model, args.prompt)
        del model.bert

        print(f"Enhancer params take {count_parameters(model.enhancer)/num_model_params*100:.2f}%")

    elif args.arch == 'head':
        model = load_model(args.config).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.transformer.train()
        for p in model.transformer.parameters():
            p.requires_grad = True

        if args.ckpt is not None:
            print('loading from', args.ckpt)
            model.transformer.load_state_dict(torch.load(args.ckpt))
        optimizer = torch.optim.Adam(model.transformer.parameters(), lr=args.lr)

        with torch.no_grad():
            text_dict_cache = get_text_dict(model, args.prompt)
        
        print(f"Head params take {count_parameters(model.transformer)/count_parameters(model)*100:.2f}%")    
        del model.bert
    
    elif args.arch == 'linear':
        model = load_model(args.config).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.transformer.decoder.bbox_embed.train()
        model.transformer.decoder.class_embed.train()
        for p in model.transformer.decoder.bbox_embed.parameters():
            p.requires_grad = True
        for p in model.transformer.decoder.class_embed.parameters():
            p.requires_grad = True

        if args.ckpt is not None:
            print('loading from', args.ckpt)
            linear_state_dict = torch.load(args.ckpt)
            model.transformer.decoder.bbox_embed.load_state_dict(linear_state_dict['bbox'])
            model.transformer.decoder.class_embed.load_state_dict(linear_state_dict['class'])
        
        linear_params = []
        linear_params.extend(model.transformer.decoder.bbox_embed.parameters())
        linear_params.extend(model.transformer.decoder.class_embed.parameters())
        optimizer = torch.optim.Adam(linear_params, lr=args.lr)

        with torch.no_grad():
            text_dict_cache = get_text_dict(model, args.prompt)        
        print(f"Linear params take {count_parameters([model.transformer.decoder.bbox_embed, model.transformer.decoder.class_embed])/count_parameters(model)*100:.2f}%")    
        del model.bert

    elif args.arch == 'bert':
        model = load_model(args.config).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.bert.train()
        for p in model.bert.parameters():
            p.requires_grad = True

        if args.ckpt is not None:
            print('loading from', args.ckpt)
            model.bert.load_state_dict(torch.load(args.ckpt))
        optimizer = torch.optim.Adam(model.bert.parameters(), lr=args.lr)
        print(f"Bert params take {count_parameters(model.bert)/count_parameters(model)*100:.2f}%")

    else:
        raise NotImplementedError
    gc.collect()
    torch.cuda.empty_cache()

    loss_fn = SetCriterion(
        matcher     = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, focal_alpha=0.25),
        focal_alpha = 0.25,
        focal_gamma = 2,
        losses      = ['labels', 'boxes']
    )
    loss_history_dict = {'loss_ce': [], 'loss_bbox': [], 'loss_giou': [], 'weighted': []}

    if args.dataset == 'ovcoco':
        images_fewshot, _ = get_ovcoco_images(args)
    elif args.dataset == 'rareplanes':
        images_fewshot, _ = get_rareplanes_images(args)
    else:
        if args.dataset == 'scenes100':
            images_train_fewshot, images_test_fewshot, _ = get_scenes100_images(args)
        if args.dataset == 'egoper':
            images_train_fewshot, images_test_fewshot, _ = get_egoper_images(args)
        if args.dataset == 'hoist':
            images_train_fewshot, images_test_fewshot, _ = get_hoist_images(args)
        if args.dataset == 'birdsai':
            images_train_fewshot, images_test_fewshot, _ = get_birdsai_images(args)
        images_fewshot = (images_train_fewshot + images_test_fewshot) if args.setting == 'seen' else images_train_fewshot
    
    print('%s %d-shot %s training set: %d images' % (args.dataset, args.shot, args.setting, len(images_fewshot)))
    prefix = os.path.join(args.savedir, 'gdino%s_%s_%dshot_%s_%s' % (args.config, args.dataset, args.shot, args.setting, args.arch))
    print(prefix)

    dst_train = BoxDataset(images_fewshot, image_transform)
    loader_train = torch.utils.data.DataLoader(dst_train, batch_size=args.image_batch_size, collate_fn=BoxDataset.collate, shuffle=True, num_workers=args.num_workers)
    iter_train = iter(loader_train)
    
    with torch.autograd.set_detect_anomaly(True):
        eval_iters = list(range(0, args.iters, args.eval_interval)) + [args.iters - 1]
        for it in tqdm.tqdm(range(0, args.iters), ascii=True):
            try:
                batch = next(iter_train)
            except StopIteration:
                iter_train = iter(loader_train)
                batch = next(iter_train)
            im_torch_batch, targets_batch, _ = batch
            optimizer.zero_grad()

            if args.arch == 'prompt':
                text_dict = text_dict_cache
            elif args.arch in ['enhancer', 'head', 'linear', 'losa', 'res_tuner']:
                text_dict = copy.deepcopy(text_dict_cache)
            else:
                text_dict = get_text_dict(model, args.prompt)

            targets_all, outputs_all = [], {'pred_boxes': [], 'pred_logits': []}
            for im_torch, targets in zip(im_torch_batch, targets_batch):
                if args.arch == 'prompt':
                    text_dict_copy = text_dict.copy()
                    text_dict_copy['encoded_text'] = text_dict['encoded_text'].clone()
                    boxes, logits = model_forward_single_image(model, text_dict_copy, im_torch.cuda(), args.arch)
                else:
                    boxes, logits = model_forward_single_image(model, text_dict, im_torch.cuda(), args.arch)
                targets['boxes'], targets['labels'] = targets['boxes'].cuda(), targets['labels'].cuda()
                targets_all.append(targets)
                outputs_all['pred_boxes'].append(boxes)
                outputs_all['pred_logits'].append(logits)
            outputs_all['pred_boxes'] = torch.stack(outputs_all['pred_boxes'], dim=0)
            outputs_all['pred_logits'] = torch.stack(outputs_all['pred_logits'], dim=0).unsqueeze(2)

            loss_dict = loss_fn(outputs_all, targets_all)
            L = loss_dict['loss_ce'] * args.cls_loss_coef + loss_dict['loss_bbox'] * args.bbox_loss_coef + loss_dict['loss_giou'] * args.giou_loss_coef
            L.backward()
            optimizer.step()

            loss_history_dict['loss_ce'].append(float(loss_dict['loss_ce'].item()))
            loss_history_dict['loss_bbox'].append(float(loss_dict['loss_bbox'].item()))
            loss_history_dict['loss_giou'].append(float(loss_dict['loss_giou'].item()))
            loss_history_dict['weighted'].append(float(L.item()))

            if it in eval_iters:
                if args.arch == 'lora':
                    lora_state_dict = {}
                    for i, lora_layer in enumerate(model.lora_layers):
                        # Save each LoRA layer's state dict
                        lora_state_dict[f"lora_{i}"] = lora_layer.state_dict()
                    torch.save(lora_state_dict, prefix + '.pth')
                elif args.arch == 'losa':
                    torch.save(model.transformer.encoder.losa.state_dict(), prefix + '.pth')
                elif args.arch == 'res_tuner':
                    res_tuner_state_dict = {}
                    for i, tuner_layer in enumerate(model.tuner_layers):
                        # Save each ResTuner layer's state dict
                        res_tuner_state_dict[f"tuner_{i}"] = tuner_layer.state_dict()
                    torch.save(res_tuner_state_dict, prefix + '.pth')
                elif args.arch == 'bitfit':
                    bias_dict = {name: param for name, param in model.named_parameters() if 'bias' in name and 'bert' in name}
                    torch.save(bias_dict, prefix + '.pth')
                elif args.arch == 'prompt':
                    torch.save(text_dict_cache['encoded_text'], prefix + '.pth')
                elif args.arch == 'enhancer':
                    torch.save(model.enhancer.state_dict(), prefix + '.pth')
                elif args.arch == 'head':
                    torch.save(model.transformer.state_dict(), prefix + '.pth')
                elif args.arch == 'linear':
                    linear_state_dict = {'bbox': model.transformer.decoder.bbox_embed.state_dict(), 'class': model.transformer.decoder.class_embed.state_dict()}
                    torch.save(linear_state_dict, prefix + '.pth')
                elif args.arch == 'bert':
                    torch.save(model.bert.state_dict(), prefix + '.pth')
                else:
                    raise NotImplementedError
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


@torch.no_grad()
def eval_fewshot(args):
    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()

    print('loading from', args.ckpt)
    if args.arch == 'lora':
        insert_lora(model, args)
        lora_state_dict = torch.load(args.ckpt)
        # Load each saved LoRA layer's state dict back into the model
        for i, lora_layer in enumerate(model.lora_layers):
            lora_layer.load_state_dict(lora_state_dict[f"lora_{i}"])
    elif args.arch == 'losa':
        insert_losa(model, args)
        model.transformer.encoder.losa.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'res_tuner':
        insert_res_tuner(model, args)
        res_tuner_state_dict = torch.load(args.ckpt)
        # Load each saved Tuner layer's state dict back into the model
        for i, tuner_layer in enumerate(model.tuner_layers):
            tuner_layer.load_state_dict(res_tuner_state_dict[f"tuner_{i}"])
    elif args.arch == 'bitfit':
        bias_dict = torch.load(args.ckpt)
        # Load each bias term back into the model
        with torch.no_grad():  # Prevents tracking in the computation graph
            for name, param in model.named_parameters():
                if name in bias_dict:
                    param.copy_(bias_dict[name])
    elif args.arch == 'prompt':
        text_dict = get_text_dict(model, args.prompt)
        text_dict['encoded_text'] = torch.load(args.ckpt)
    elif args.arch == 'enhancer':
        model.enhancer = PromptEnhancer(args.enhancer_r_qkv, args.enhancer_r_ff).cuda()
        model.enhancer.eval()
        model.enhancer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'head':
        model.transformer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'linear':
        linear_state_dict = torch.load(args.ckpt)
        model.transformer.decoder.bbox_embed.load_state_dict(linear_state_dict['bbox'])
        model.transformer.decoder.class_embed.load_state_dict(linear_state_dict['class'])
    elif args.arch == 'bert':
        model.bert.load_state_dict(torch.load(args.ckpt))
    else:
        raise NotImplementedError

    if args.arch != 'prompt':
        text_dict = get_text_dict(model, args.prompt)

    if args.dataset == 'ovcoco':
        _, images_eval = get_ovcoco_images(args)
    if args.dataset == 'rareplanes':
        _, images_eval = get_rareplanes_images(args)
    if args.dataset == 'scenes100':
        _, _, images_eval = get_scenes100_images(args)
    if args.dataset == 'egoper':
        _, _, images_eval = get_egoper_images(args)
    if args.dataset == 'hoist':
        _, _, images_eval = get_hoist_images(args)
    if args.dataset == 'birdsai':
        _, _, images_eval = get_birdsai_images(args)
    loader = torch.utils.data.DataLoader(
        BoxDataset(images_eval, image_transform, training=False),
        batch_size=None, shuffle=False, num_workers=4
    )

    detections = []
    for im_torch, im in tqdm.tqdm(loader, ascii=True, total=len(images_eval)):
        im['annotations'] = []
        boxes, logits = model_forward_single_image(model, copy.deepcopy(text_dict), im_torch.cuda(), args.arch)
        scores = logits.sigmoid()
        filt_mask = scores > args.box_threshold
        boxes, scores = boxes[filt_mask], scores[filt_mask]
        for box, s in zip(boxes, scores):
            xc, yc, w, h = map(float, [box[0] * im['width'], box[1] * im['height'], box[2] * im['width'], box[3] * im['height']])
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': 0, 'score': float(s), 'bbox_mode': 0})
        detections.append(im)

    if args.dataset == 'scenes100':
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs = evaluate_scenes100_masked_fewshot(images_eval, detections)
    else:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs = eval_AP(images_eval, detections, return_thres=False)
    print(APs['results'])


@torch.no_grad()
def show_detected(args):
    import math
    import skimage.io
    import imantics
    from evaluation import _check_overlap

    def draw_bbox(im, annotations, title, font, lw):
        im = Image.fromarray(im, 'RGB')
        draw = ImageDraw.Draw(im)
        draw.text((20, 20), title, fill='#000000', stroke_width=3, font=font)
        draw.text((20, 20), title, fill='#FFFFFF', stroke_width=1, font=font)
        for ann in annotations:
            # bbox has format [x1, y1, x2, y2]
            x1, y1, x2, y2 = ann['bbox']
            draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='#FF0000', width=lw)
        im = np.array(im)
        return im

    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()

    print('loading from', args.ckpt)
    if args.arch == 'lora':
        insert_lora(model, args)
        lora_state_dict = torch.load(args.ckpt)
        # Load each saved LoRA layer's state dict back into the model
        for i, lora_layer in enumerate(model.lora_layers):
            lora_layer.load_state_dict(lora_state_dict[f"lora_{i}"])
    elif args.arch == 'losa':
        insert_losa(model, args)
        model.transformer.encoder.losa.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'res_tuner':
        insert_res_tuner(model, args)
        res_tuner_state_dict = torch.load(args.ckpt)
        # Load each saved Tuner layer's state dict back into the model
        for i, tuner_layer in enumerate(model.tuner_layers):
            tuner_layer.load_state_dict(res_tuner_state_dict[f"tuner_{i}"])
    elif args.arch == 'bitfit':
        bias_dict = torch.load(args.ckpt)
        # Load each bias term back into the model
        with torch.no_grad():  # Prevents tracking in the computation graph
            for name, param in model.named_parameters():
                if name in bias_dict:
                    param.copy_(bias_dict[name])
    elif args.arch == 'prompt':
        text_dict = get_text_dict(model, args.prompt)
        text_dict['encoded_text'] = torch.load(args.ckpt)
    elif args.arch == 'enhancer':
        model.enhancer = PromptEnhancer(args.enhancer_r_qkv, args.enhancer_r_ff).cuda()
        model.enhancer.eval()
        model.enhancer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'head':
        model.transformer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'linear':
        linear_state_dict = torch.load(args.ckpt)
        model.transformer.decoder.bbox_embed.load_state_dict(linear_state_dict['bbox'])
        model.transformer.decoder.class_embed.load_state_dict(linear_state_dict['class'])
    elif args.arch == 'bert':
        model.bert.load_state_dict(torch.load(args.ckpt))
    else:
        raise NotImplementedError
        
    if args.arch != 'prompt':
        text_dict = get_text_dict(model, args.prompt)

    if args.dataset == 'ovcoco':
        _, images_eval = get_ovcoco_images(args)
    if args.dataset == 'rareplanes':
        _, images_eval = get_rareplanes_images(args)
    if args.dataset == 'scenes100':
        _, _, images_eval = get_scenes100_images(args)
    if args.dataset == 'egoper':
        _, _, images_eval = get_egoper_images(args)
    if args.dataset == 'hoist':
        _, _, images_eval = get_hoist_images(args)
        images_eval = list(filter(lambda x: len(x['annotations']) > 0, images_eval))
        video_id_list = {}
        for im in images_eval:
            if not im['video_id'] in video_id_list:
                video_id_list[im['video_id']] = 0
            video_id_list[im['video_id']] += 1
        video_id_list = [v for v in video_id_list if video_id_list[v] >= 4]
        video_id_list = [video_id_list[i] for i in range(0, len(video_id_list), len(video_id_list) // 50)]
        images_eval = list(filter(lambda x: x['video_id'] in video_id_list, images_eval))

    loader = torch.utils.data.DataLoader(
        BoxDataset(images_eval, image_transform, training=False),
        batch_size=None, shuffle=False, num_workers=4
    )
    detections_per_video, images_per_video = {}, {}
    for im_torch, im in tqdm.tqdm(loader, ascii=True, total=len(images_eval)):
        if not im['video_id'] in images_per_video:
            images_per_video[im['video_id']] = []
            detections_per_video[im['video_id']] = []
        images_per_video[im['video_id']].append(copy.deepcopy(im))
        im['annotations'] = []
        boxes, logits = model_forward_single_image(model, copy.deepcopy(text_dict), im_torch.cuda(), args.arch)
        scores = logits.sigmoid()
        for box, s in zip(boxes, scores):
            xc, yc, w, h = map(float, [box[0] * im['width'], box[1] * im['height'], box[2] * im['width'], box[3] * im['height']])
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': 0, 'score': float(s), 'bbox_mode': 0})
        detections_per_video[im['video_id']].append(im)

    if args.dataset == 'scenes100':
        mask_rgb, mask_alpha = [0, 1, 0], 0.3
        im_arr_templates = {}
        for video_id in images_per_video:
            im_arr_templates[video_id] = images_per_video[video_id][-1]['file_name']
        im_arr_templates = {k: skimage.io.imread(v) for k, v in im_arr_templates.items()}
        with open(os.path.join(os.path.dirname(__file__), '..', '..', 'masks.json'), 'r') as fp:
            masks = json.load(fp)
        masks = {m['video']: m['polygons'] for m in masks if m['video'] in im_arr_templates}
        for video_id in masks:
            if len(masks[video_id]) > 0:
                m_arr = imantics.Annotation.from_polygons(masks[video_id], image=imantics.Image(im_arr_templates[video_id]))
                m_arr = np.expand_dims(m_arr.array.astype(np.float16), 2) * mask_alpha
            else:
                m_arr = None
            masks[video_id] = m_arr
        for video_id in images_per_video:
            for im in images_per_video[video_id] + detections_per_video[video_id]:
                annotations_filter = []
                for ann in im['annotations']:
                    if ann['bbox_mode'] == 1:
                        x1, y1, w, h = ann['bbox']
                        x2, y2 = x1 + w, y1 + h
                        ann['bbox_mode'] = 0
                        ann['bbox'] = [x1, y1, x2, y2]
                    if ann['bbox_mode'] != 0:
                        raise 'box modes unrecognized'
                    if not _check_overlap(masks[im['video_id']], ann['bbox']):
                        annotations_filter.append(ann)
                im['annotations'] = annotations_filter

    outputdir = './case_study'
    for video_id in images_per_video:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            _images, _detections = images_per_video[video_id], detections_per_video[video_id]
            _images, _detections = map(lambda x: sorted(x, key=lambda y: y['file_name']), [_images, _detections])
            APs = eval_AP(_images, _detections, return_thres=False)
        print(video_id, APs['results']['person'])

        pr = np.array(APs['raw']['precision'])[5, :, :, 0, 2].T
        rc = np.array(APs['raw']['recThrs'])
        s  = np.array(APs['raw']['scores'])[5, :, :, 0, 2].T
        f1 = 2 * pr[0] * rc / (pr[0] + rc)
        t = s[0][np.argmax(f1)]
        # print(f1, f1.max(), t)

        _WH = max(_images[-1]['width'], _images[-1]['height'])
        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=int(_WH / 40))
        im = skimage.io.imread(_images[-1]['file_name'])
        pred = list(filter(lambda x: x['score'] > t, _detections[-1]['annotations']))
        skimage.io.imsave(os.path.join(outputdir, '%s_%s_unseen.jpg' % (args.dataset, video_id)), draw_bbox(im, pred, 'Adapted', font, math.ceil(_WH / 500)), quality=90)

@torch.no_grad()
def exclusive_test_scenes100(args):
    import math
    import skimage.io
    import imantics
    from evaluation import _check_overlap

    def draw_bbox(im, annotations, title, font, lw):
        im = Image.fromarray(im, 'RGB')
        draw = ImageDraw.Draw(im)
        draw.text((20, 20), title, fill='#000000', stroke_width=3, font=font)
        draw.text((20, 20), title, fill='#FFFFFF', stroke_width=1, font=font)
        for ann in annotations:
            # bbox has format [x1, y1, x2, y2]
            x1, y1, x2, y2 = ann['bbox']
            draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='#FF0000', width=lw)
        im = np.array(im)
        return im

    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()

    print('loading from', args.ckpt)
    if args.arch == 'lora':
        insert_lora(model, args)
        lora_state_dict = torch.load(args.ckpt)
        # Load each saved LoRA layer's state dict back into the model
        for i, lora_layer in enumerate(model.lora_layers):
            lora_layer.load_state_dict(lora_state_dict[f"lora_{i}"])
    elif args.arch == 'losa':
        insert_losa(model, args)
        model.transformer.encoder.losa.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'res_tuner':
        insert_res_tuner(model, args)
        res_tuner_state_dict = torch.load(args.ckpt)
        # Load each saved Tuner layer's state dict back into the model
        for i, tuner_layer in enumerate(model.tuner_layers):
            tuner_layer.load_state_dict(res_tuner_state_dict[f"tuner_{i}"])
    elif args.arch == 'bitfit':
        bias_dict = torch.load(args.ckpt)
        # Load each bias term back into the model
        with torch.no_grad():  # Prevents tracking in the computation graph
            for name, param in model.named_parameters():
                if name in bias_dict:
                    param.copy_(bias_dict[name])
    elif args.arch == 'prompt':
        text_dict = get_text_dict(model, args.prompt)
        text_dict['encoded_text'] = torch.load(args.ckpt)
    elif args.arch == 'enhancer':
        model.enhancer = PromptEnhancer(args.enhancer_r_qkv, args.enhancer_r_ff).cuda()
        model.enhancer.eval()
        model.enhancer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'head':
        model.transformer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'linear':
        linear_state_dict = torch.load(args.ckpt)
        model.transformer.decoder.bbox_embed.load_state_dict(linear_state_dict['bbox'])
        model.transformer.decoder.class_embed.load_state_dict(linear_state_dict['class'])
    elif args.arch == 'bert':
        model.bert.load_state_dict(torch.load(args.ckpt))
    else:
        raise NotImplementedError
    
    if args.arch != 'prompt':
        text_dict = get_text_dict(model, args.prompt)

    _, _, images_eval = get_scenes100_images(args, keep_class=True)
  
    loader = torch.utils.data.DataLoader(
        BoxDataset(images_eval, image_transform, training=False),
        batch_size=None, shuffle=False, num_workers=4
    )
    detections = []
    detections_per_video, images_per_video = {}, {}
    for im_torch, im in tqdm.tqdm(loader, ascii=True, total=len(images_eval)):
        if not im['video_id'] in images_per_video:
            images_per_video[im['video_id']] = []
            detections_per_video[im['video_id']] = []
        images_per_video[im['video_id']].append(copy.deepcopy(im))
        im['annotations'] = []
        boxes, logits = model_forward_single_image(model, copy.deepcopy(text_dict), im_torch.cuda(), args.arch)
        scores = logits.sigmoid()
        filt_mask = scores > args.box_threshold
        boxes, scores = boxes[filt_mask], scores[filt_mask]
        for box, s in zip(boxes, scores):
            xc, yc, w, h = map(float, [box[0] * im['width'], box[1] * im['height'], box[2] * im['width'], box[3] * im['height']])
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': args.inclusive_class, 'score': float(s), 'bbox_mode': 0})
        detections_per_video[im['video_id']].append(im)
        detections.append(im)
    
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        APs = evaluate_scenes100_masked_fewshot(images_eval, detections)

    print(APs['results'])
    
    # remove detections in masked area
    mask_rgb, mask_alpha = [0, 1, 0], 0.3
    im_arr_templates = {}
    for video_id in images_per_video:
        im_arr_templates[video_id] = images_per_video[video_id][-1]['file_name']
    im_arr_templates = {k: skimage.io.imread(v) for k, v in im_arr_templates.items()}
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'masks.json'), 'r') as fp:
        masks = json.load(fp)
    masks = {m['video']: m['polygons'] for m in masks if m['video'] in im_arr_templates}
    for video_id in masks:
        if len(masks[video_id]) > 0:
            m_arr = imantics.Annotation.from_polygons(masks[video_id], image=imantics.Image(im_arr_templates[video_id]))
            m_arr = np.expand_dims(m_arr.array.astype(np.float16), 2) * mask_alpha
        else:
            m_arr = None
        masks[video_id] = m_arr
    for video_id in images_per_video:
        for im in images_per_video[video_id] + detections_per_video[video_id]:
            annotations_filter = []
            for ann in im['annotations']:
                if ann['bbox_mode'] == 1:
                    x1, y1, w, h = ann['bbox']
                    x2, y2 = x1 + w, y1 + h
                    ann['bbox_mode'] = 0
                    ann['bbox'] = [x1, y1, x2, y2]
                if ann['bbox_mode'] != 0:
                    raise 'box modes unrecognized'
                if not _check_overlap(masks[im['video_id']], ann['bbox']):
                    annotations_filter.append(ann)
            im['annotations'] = annotations_filter

    for video_id in images_per_video:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            _images, _detections = images_per_video[video_id], detections_per_video[video_id]
            _images, _detections = map(lambda x: sorted(x, key=lambda y: y['file_name']), [_images, _detections])
            APs = eval_AP(_images, _detections, return_thres=False)
        # print(video_id, APs['results'])

        pr = np.array(APs['raw']['precision'])[5, :, :, 0, 2].T
        rc = np.array(APs['raw']['recThrs'])
        s  = np.array(APs['raw']['scores'])[5, :, :, 0, 2].T
        f1 = 2 * pr[args.inclusive_class] * rc / (pr[args.inclusive_class] + rc)
        t = s[args.inclusive_class][np.argmax(f1)]
        # print(f1, f1.max(), t)

        _WH = max(_images[-1]['width'], _images[-1]['height'])
        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=int(_WH / 40))
        im = skimage.io.imread(_images[-1]['file_name'])
        pred = list(filter(lambda x: x['score'] > t, _detections[-1]['annotations']))
        skimage.io.imsave(os.path.join(args.savedir, '%s_%s_unseen.jpg' % (args.dataset, video_id)), draw_bbox(im, pred, 'Unseen Adapted', font, math.ceil(_WH / 500)), quality=90)



@torch.no_grad()
def inference_throughput(args):
    import gc
    import time

    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
    if args.arch == 'lora':
        insert_lora(model, args)
        lora_state_dict = torch.load(args.ckpt)
        # Load each saved LoRA layer's state dict back into the model
        for i, lora_layer in enumerate(model.lora_layers):
            lora_layer.load_state_dict(lora_state_dict[f"lora_{i}"])
    elif args.arch == 'losa':
        insert_losa(model, args)
        model.transformer.encoder.losa.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'res_tuner':
        insert_res_tuner(model, args)
        res_tuner_state_dict = torch.load(args.ckpt)
        # Load each saved Tuner layer's state dict back into the model
        for i, tuner_layer in enumerate(model.tuner_layers):
            tuner_layer.load_state_dict(res_tuner_state_dict[f"tuner_{i}"])
    elif args.arch == 'prompt':
        text_dict = get_text_dict(model, args.prompt)
        text_dict['encoded_text'] = torch.load(args.ckpt)
    elif args.arch == 'enhancer':
        model.enhancer = PromptEnhancer(args.enhancer_r_qkv, args.enhancer_r_ff).cuda()
        model.enhancer.eval()
        model.enhancer.load_state_dict(torch.load(args.ckpt))
    
    if args.arch != 'prompt':
        text_dict = get_text_dict(model, args.prompt)
    print(args.prompt, text_dict['tokenized_input_ids'].size())

    if args.dataset == 'scenes100':
        _, _, images_eval = get_scenes100_images(args)
    if args.dataset == 'egoper':
        _, _, images_eval = get_egoper_images(args)
    if args.dataset == 'hoist':
        _, _, images_eval = get_hoist_images(args)
    im = images_eval[-1]
    im_torch, _ = image_transform(Image.open(im['file_name']).convert('RGB'), None)
    im_torch = im_torch.cuda()
    print(im['file_name'], im_torch.size(), im_torch.dtype)
    gc.collect()
    torch.cuda.empty_cache()

    N1, N2 = 100, 400
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            text_dict = get_text_dict(model, args.prompt)
            boxes, logits = model_forward_single_image(model, text_dict, im_torch, args.arch)
            scores = logits.sigmoid()
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Grounding DINO example', add_help=True)
    parser.add_argument('--opt', type=str)
    parser.add_argument('--config', '-c', type=str, default='swint', choices=['swint', 'swinb'], help='path to config file')
    parser.add_argument('--box_threshold', type=float, default=0.05, help='box threshold')

    parser.add_argument('--dataset', type=str, default=None, choices=['scenes100', 'egoper', 'hoist', 'ovcoco', 'birdsai', 'rareplanes'])
    parser.add_argument('--cocodir', type=str, default='../../MSCOCO2017')
    parser.add_argument('--birdsai_dir', type=str, default='../../../BIRDSAI')
    parser.add_argument('--egoper_dir', type=str, default='../../../PTG/ptg_detection')
    parser.add_argument('--hoist_dir', type=str, default='../../../OIH_VIS')
    parser.add_argument('--rareplanes_dir', type=str, default='../../../RarePlanes')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--inclusive_class', type=int, default=0, choices=[0, 1])
    parser.add_argument('--shot', type=int, default=1, help='few-shot learning')
    parser.add_argument('--setting', type=str, default='unseen', choices=['unseen', 'seen'])

    parser.add_argument('--arch', type=str, default='enhancer', choices=['enhancer', 'head', 'bert', 'lora', 'losa', 'res_tuner', 'linear', 'bitfit', 'prompt'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--cls_loss_coef', type=float, default=2.0)
    parser.add_argument('--bbox_loss_coef', type=float, default=5.0)
    parser.add_argument('--giou_loss_coef', type=float, default=2.0)

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=2, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--savedir', type=str, default='.')

    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=1/16, help='LoRA alpha coefficient')
    parser.add_argument('--lora_query', type=bool, default=True, help='make LoRA in query embedding or not')
    parser.add_argument('--lora_key', type=bool, default=True, help='make LoRA in key embedding or not')
    parser.add_argument('--lora_value', type=bool, default=True, help='make LoRA in value embedding or not')
    
    parser.add_argument('--losa_r', type=int, default=16, help='LoSA rank')
    parser.add_argument('--res_tuner_r', type=int, default=16, help='ResTuner rank')
    parser.add_argument('--enhancer_r_qkv', type=int, default=16, help='Enhancer rank QKV')
    parser.add_argument('--enhancer_r_ff', type=int, default=16, help='Enhancer rank Feed Forward')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.opt == 'train':
        train_few_shot(args)
    if args.opt == 'eval':
        eval_fewshot(args)
    if args.opt == 'case_study':
        show_detected(args)
    if args.opt == 'tp':
        inference_throughput(args)
    if args.opt == 'ext':
        exclusive_test_scenes100(args)

'''
python finetune_groundingdino_accv.py --opt train --iters 2000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --arch enhancer --dataset scenes100 --shot 1 --prompt "person . vehicle ." --setting seen
python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset scenes100 --shot 1 --prompt "person . vehicle ." --ckpt gdinoswint_scenes100_1shot_seen_enhancer.pth

python finetune_groundingdino_accv.py --opt train --iters 2000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --arch head --dataset scenes100 --shot 1 --prompt "person . vehicle ." --setting seen
python finetune_groundingdino_accv.py --opt eval --arch head --dataset scenes100 --shot 1 --prompt "person . vehicle ." --ckpt gdinoswint_scenes100_1shot_seen_head.pth

python finetune_groundingdino_accv.py --opt train --iters 2000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --arch bert --dataset scenes100 --shot 1 --prompt "person . vehicle ." --setting seen
python finetune_groundingdino_accv.py --opt eval --arch bert --dataset scenes100 --shot 1 --prompt "person . vehicle ." --ckpt gdinoswint_scenes100_1shot_seen_bert.pth


python finetune_groundingdino_accv.py --opt train --iters 2000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --arch enhancer --dataset egoper --shot 1 --prompt "kitchen . cooking . human body ." --setting seen
python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset egoper --shot 1 --prompt "kitchen . cooking . human body ." --ckpt gdinoswint_egoper_1shot_seen_enhancer.pth

python finetune_groundingdino_accv.py --opt train --iters 2000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --arch head --dataset egoper --shot 1 --prompt "kitchen . cooking . human body ." --setting seen
python finetune_groundingdino_accv.py --opt eval --arch head --dataset egoper --shot 1 --prompt "kitchen . cooking . human body ." --ckpt gdinoswint_egoper_1shot_seen_head.pth

python finetune_groundingdino_accv.py --opt train --iters 2000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --arch bert --dataset egoper --shot 1 --prompt "kitchen . cooking . human body ." --setting seen
python finetune_groundingdino_accv.py --opt eval --arch bert --dataset egoper --shot 1 --prompt "kitchen . cooking . human body ." --ckpt gdinoswint_egoper_1shot_seen_bert.pth


python finetune_groundingdino_accv.py --opt train --iters 20000 --eval_interval 5001 --image_batch_size 2 --num_workers 2 --arch enhancer --dataset hoist --shot 1 --prompt "object in hand ." --setting seen
python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset hoist --shot 1 --prompt "object in hand ." --ckpt gdinoswint_hoist_1shot_seen_enhancer.pth

python finetune_groundingdino_accv.py --opt train --iters 20000 --eval_interval 5001 --image_batch_size 2 --num_workers 2 --arch head --dataset hoist --shot 1 --prompt "object in hand ." --setting seen
python finetune_groundingdino_accv.py --opt eval --arch head --dataset hoist --shot 1 --prompt "object in hand ." --ckpt gdinoswint_hoist_1shot_seen_head.pth

python finetune_groundingdino_accv.py --opt train --iters 20000 --eval_interval 5001 --image_batch_size 2 --num_workers 2 --arch bert --dataset hoist --shot 1 --prompt "object in hand ." --setting seen
python finetune_groundingdino_accv.py --opt eval --arch bert --dataset hoist --shot 1 --prompt "object in hand ." --ckpt gdinoswint_hoist_1shot_seen_bert.pth

python finetune_groundingdino_accv.py --opt train --iters 10000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --arch enhancer --dataset ovcoco --shot 1 --prompt "coco objects ." 
python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset ovcoco --shot 1 --prompt "coco objects ." --ckpt gdinoswint_ovcoco_1shot_seen_enhancer.pth

python finetune_groundingdino_accv.py --opt train --iters 2000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --arch enhancer --dataset birdsai --shot 1 --prompt "nighttime images of animals and humans ." --savedir birdsai_p1
python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset birdsai --shot 1 --prompt "nighttime images of animals and humans ." --ckpt gdinoswint_birdsai_1shot_seen_enhancer.pth
thermal infrared objects .
animals . humans .

python finetune_groundingdino_accv.py --opt train --iters 2000 --eval_interval 501 --image_batch_size 2 --num_workers 2 --arch enhancer --dataset rareplanes --shot 1 --prompt "planes ." --savedir rareplanes_p1
python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset rareplanes --shot 1 --prompt "planes ." --ckpt gdinoswint_rareplanes_1shot_seen_enhancer.pth
aircrafts .
airplane . aeroplane . airliner . aircraft .



python finetune_groundingdino_accv.py --opt case_study --arch enhancer --dataset scenes100 --shot 1 --prompt "person . vehicle ." --ckpt /mnt/f/intersections_results/accv24/gdino/Scenes100_person_vehicle/gdinoswint_scenes100_1shot_unseen_enhancer.pth

python finetune_groundingdino_accv.py --opt case_study --arch enhancer --dataset egoper --shot 1 --prompt "kitchen object ." --ckpt /mnt/f/intersections_results/accv24/gdino/EgoPER_kitchenobject/gdinoswint_egoper_1shot_unseen_enhancer.pth

python finetune_groundingdino_accv.py --opt case_study --arch enhancer --dataset hoist --shot 1 --prompt "hand-held object ." --ckpt /mnt/f/intersections_results/accv24/gdino/HOIST_hand-held_object/gdinoswint_hoist_1shot_unseen_enhancer_iter_19999.pth


cp /mnt/f/intersections_results/accv24/gdino/EgoPER_kitchenobject/gdinoswint_egoper_1shot_unseen_enhancer.pth /tmp
python finetune_groundingdino_accv.py --opt tp --arch enhancer --shot 1 --dataset egoper --ckpt /tmp/gdinoswint_egoper_1shot_unseen_enhancer.pth --prompt "kitchen object ."
python finetune_groundingdino_accv.py --opt tp --arch head --shot 1 --dataset egoper --prompt "kitchen object ."
python finetune_groundingdino_accv.py --opt tp --arch enhancer --shot 1 --dataset egoper --ckpt /tmp/gdinoswint_egoper_1shot_unseen_enhancer.pth --prompt "kitchen . cooking . human body ."
python finetune_groundingdino_accv.py --opt tp --arch head --shot 1 --dataset egoper --prompt "kitchen . cooking . human body ."
python finetune_groundingdino_accv.py --opt tp --arch enhancer --shot 1 --dataset egoper --ckpt /tmp/gdinoswint_egoper_1shot_unseen_enhancer.pth --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ."
python finetune_groundingdino_accv.py --opt tp --arch head --shot 1 --dataset egoper --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ."
python finetune_groundingdino_accv.py --opt tp --arch enhancer --shot 1 --dataset egoper --ckpt /tmp/gdinoswint_egoper_1shot_unseen_enhancer.pth --prompt "kettle . measuring cup . kitchen scale . coffee grinder . filter cone dripper . paper basket filter . mug . thermometer . spoon . bowl . bottle . hand . paper filter . tortilla . peanut butter . jam . toothpick . knife . chopping board . dental floss . paper towel . plate . paper sheet . tea bag . trash can . honey . nutella . banana slice . cinnamon . oat . microwave . raisin ."
python finetune_groundingdino_accv.py --opt tp --arch head --shot 1 --dataset egoper --prompt "kettle . measuring cup . kitchen scale . coffee grinder . filter cone dripper . paper basket filter . mug . thermometer . spoon . bowl . bottle . hand . paper filter . tortilla . peanut butter . jam . toothpick . knife . chopping board . dental floss . paper towel . plate . paper sheet . tea bag . trash can . honey . nutella . banana slice . cinnamon . oat . microwave . raisin ."

python finetune_groundingdino_accv.py --opt ext --arch enhancer --dataset scenes100 --shot 1 --prompt "person ." --inclusive_class 0 --ckpt gdinoswint_scenes100_1shot_unseen_enhancer.pth --savedir ./case_study/scenes100/person/
'''

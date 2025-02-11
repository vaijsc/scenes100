import argparse
import os
import sys
import json
import tqdm
import copy
import contextlib
import random
import csv
from functools import partial 

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Optional

groundingdino_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GroundingDINO')
sys.path.append(groundingdino_dir)
import groundingdino.datasets.transforms as T
from groundingdino.models.GroundingDINO.transformer import TransformerEncoder, TransformerDecoder, DeformableTransformerEncoderLayer
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from groundingdino.models.GroundingDINO.fuse_modules import BiAttentionBlock
from groundingdino.models.GroundingDINO.ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from groundingdino.models.GroundingDINO.utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation import eval_AP

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # ERROR: huggingface/tokenizers: The current process just got forked, after parallelism has already been used.

def count_parameters(modules, include_non_trainable=True):
    # Convert single module to a list for consistent processing
    if isinstance(modules, torch.nn.Module):
        modules = [modules]
    
    # Define the counting function
    return sum(
        p.numel()
        for module in modules
        for p in module.parameters()
        if include_non_trainable or p.requires_grad
    )


def load_model(config):
    if config == 'swint':
        config_file = os.path.join(groundingdino_dir, 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
        checkpoint_path = os.path.join(groundingdino_dir, 'groundingdino_swint_ogc.pth')
    else:
        assert config == 'swinb'
        config_file = os.path.join(groundingdino_dir, 'groundingdino', 'config', 'GroundingDINO_SwinB.cfg.py')
        checkpoint_path = os.path.join(groundingdino_dir, 'groundingdino_swinb_cogcoor.pth')
    args = SLConfig.fromfile(config_file)
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    print('backbone:   ', args.backbone)
    print('enc_layers: ', args.enc_layers)
    print('dec_layers: ', args.dec_layers)
    print('num_queries:', args.num_queries)
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_res = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(load_res)
    return model


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.cuda()
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        self.cuda()
    def forward(self, x):
        return self.linear(x) + self.lora(x)


def insert_lora(model, args):
    lora_layers = []
    assign_lora = partial(LinearWithLoRA, rank=args.lora_r, alpha=args.lora_alpha)
    for layer in model.bert.encoder.layer:
        if args.lora_query:
            layer.attention.self.query = assign_lora(layer.attention.self.query)
            lora_layers.append(layer.attention.self.query.lora)
        if args.lora_key:
            layer.attention.self.key = assign_lora(layer.attention.self.key)
            lora_layers.append(layer.attention.self.key.lora)
        if args.lora_value:
            layer.attention.self.value = assign_lora(layer.attention.self.value)
            lora_layers.append(layer.attention.self.value.lora)
    for layer in model.backbone[0].layers:
        for block in layer.blocks:
            if args.lora_query or args.lora_key or args.lora_value:
                block.attn.qkv = assign_lora(block.attn.qkv)
                lora_layers.append(block.attn.qkv.lora)
    
    model.lora_layers = lora_layers


# class LoSALayer(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, rank):
#         super().__init__()
#         std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
#         self.down_proj = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
#         self.up_proj = torch.nn.Parameter(torch.zeros(rank, out_dim))
#         self.alpha = torch.nn.Parameter(torch.tensor(0.0)) 
#         self.gelu = torch.nn.GELU()
#         self.cuda()
        
#     def forward(self, x):
#         return self.gelu(x @ self.down_proj) @ self.up_proj * self.alpha 

# class LoSAParallelNetwork(torch.nn.Module):
#     def __init__(self, dims, rank):
#         super().__init__()
#         self.layers = torch.nn.ModuleList()
#         for dim in dims:
#             self.layers.append(LoSALayer(dim, dim, rank))
#         self.cuda()
#     def forward(self, features):
#         assert len(features) == len(self.layers), "wrong number of features."
#         out = features[-1]
#         for i, layer in enumerate(self.layers):
#             if i % 2 == 0 and i > 0:
#                 features[i] = features[i].transpose(1, 2)
#             out = out + features[i]
#             if i > 0:
#                 out = out.transpose(1, 2)
#             out = layer(out) 
#         # return out + features[-1]
#         return out.transpose(1, 2) + features[-1]


class LoSALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.down_proj = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.up_proj = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = torch.nn.Parameter(torch.tensor(1/16)) 
        self.gelu = torch.nn.GELU()
        self.cuda()
        
    def forward(self, x):
        return self.gelu(x @ self.down_proj) @ self.up_proj * self.alpha 

class LoSAParallelNetwork(torch.nn.Module):
    def __init__(self, dims, rank):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for dim in dims:
            self.layers.append(LoSALayer(dim, dim, rank))
        self.cuda()
    def forward(self, features):
        assert len(features) == len(self.layers), "wrong number of features."
        out = features[-1]
        for i, layer in enumerate(self.layers):
            # if i % 2 == 0 and i > 0:
            #     features[i] = features[i].transpose(1, 2)
            combined = out + features[i]
            # if i > 0:
                # combined = combined.transpose(1, 2)
                # out = out.transpose(1, 2)
            out = layer(combined) + out
        return out
        # return out.transpose(1, 2)

class TransformerEncoderWithLoSA(TransformerEncoder):
    @classmethod
    def add_losa(cls, encoder, rank):
        # dims = [256, 20906] * 3
        dims = [256] * 6
        # cls.encoder = encoder
        encoder.losa = LoSAParallelNetwork(dims, rank)
        encoder.__class__ = cls
        return encoder

    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=256, exchange_xy=False
                )

        outputs = []
        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask,
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )
            outputs.append(output)
        # breakpoint()
        output = self.losa(outputs)   
        return output, memory_text

def insert_losa(model, args):
    model.transformer.encoder = TransformerEncoderWithLoSA.add_losa(model.transformer.encoder, args.losa_r)


class ResidualAdapter(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.down_proj = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.up_proj = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = torch.nn.Parameter(torch.tensor(1/16)) 
        self.gelu = torch.nn.GELU()
        self.cuda()
        
    def forward(self, x):
        return self.gelu(x @ self.down_proj) @ self.up_proj * self.alpha

class EncoderLayerWithResAdapt(DeformableTransformerEncoderLayer):
    @classmethod
    def add_res_tuner(cls, encoder_layer, rank):
        # dims = [256, 20906] * 3
        dim = encoder_layer.self_attn.embed_dim
        encoder_layer.attn_tuner = ResidualAdapter(dim, dim, rank)
        encoder_layer.ffn_tuner = ResidualAdapter(dim, dim, rank)
        encoder_layer.__class__ = cls
        return encoder_layer

    def forward(
        self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None
    ):
        attn_res = self.attn_tuner(src)
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + attn_res
        # ffn
        src = self.forward_ffn(src) + self.ffn_tuner(src)

        return src

class EncoderWithResAdapt(torch.nn.Module):
    def __init__(self, encoder, rank):
        super().__init__()
        self.encoder = encoder
        self.block_tuner = ResidualAdapter(encoder.d_model, encoder.d_model, rank)
    
    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ):
        output, memory_text = self.encoder(src, pos, spatial_shapes, level_start_index, valid_ratios, key_padding_mask, memory_text, text_attention_mask, pos_text, text_self_attention_masks, position_ids)
        return self.block_tuner(src) + output, memory_text


def insert_res_tuner(model, args):
    tuner_layers = []
    for layer in model.transformer.encoder.layers:
        layer = EncoderLayerWithResAdapt.add_res_tuner(layer, args.res_tuner_r)
        tuner_layers.extend([layer.attn_tuner, layer.ffn_tuner])
    model.transformer.encoder = EncoderWithResAdapt(model.transformer.encoder, args.res_tuner_r)
    tuner_layers.append(model.transformer.encoder.block_tuner)
    model.tuner_layers = tuner_layers

import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rank, dropout=0.0, batch_first=False):
        super(LowRankMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.rank = rank
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Low-rank projection for Q, K, and V
        self.q_proj_low = nn.Linear(embed_dim, rank)
        self.q_proj_high = nn.Linear(rank, embed_dim)
        self.k_proj_low = nn.Linear(embed_dim, rank)
        self.k_proj_high = nn.Linear(rank, embed_dim)
        self.v_proj_low = nn.Linear(embed_dim, rank)
        self.v_proj_high = nn.Linear(rank, embed_dim)

        # Output projection
        self.out_proj_low = nn.Linear(embed_dim, rank)
        self.out_proj_high = nn.Linear(rank, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        if self.batch_first:
            # Swap batch and sequence dimensions if batch_first
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        
        # Low-rank transformation
        q = self.q_proj_high(self.q_proj_low(query))
        k = self.k_proj_high(self.k_proj_low(key))
        v = self.v_proj_high(self.v_proj_low(value))

        # Split into multiple heads
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, self.embed_dim)

        # Final output projection
        attn_output = self.out_proj_high(self.out_proj_low(attn_output))

        if self.batch_first:
            # Swap batch and sequence dimensions back
            attn_output = attn_output.transpose(0, 1)

        return attn_output


class LowRankTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, rank=16, dropout=0.1, batch_first=False):
        super(LowRankTransformerDecoderLayer, self).__init__()
        self.batch_first = batch_first
        self.self_attn = LowRankMultiheadAttention(d_model, n_heads, rank, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = LowRankMultiheadAttention(d_model, n_heads, rank, dropout=dropout, batch_first=batch_first)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer norms and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        
        if self.batch_first:
            # Transpose the batch and sequence dimensions if batch_first
            tgt = tgt.transpose(0, 1)
            memory = memory.transpose(0, 1)

        # Self-attention on target with padding mask
        tgt2 = self.self_attn(tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Multi-head attention with encoder memory and padding mask
        tgt2 = self.multihead_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward network
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if self.batch_first:
            # Transpose back to batch-first format
            tgt = tgt.transpose(0, 1)

        return tgt


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption = caption + '.'
    outputs = model(image[None], captions=[caption])
    logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append([pred_phrase.split(' '), float(logit.max().item())])
    return boxes_filt, pred_phrases


def get_text_dict(model, caption):
    # print('prompts:', caption)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption = caption + '.'
    # encoder texts
    tokenized = model.tokenizer([caption], padding='longest', return_tensors='pt').to('cuda')
    text_self_attention_masks, position_ids, cate_to_token_mask_list = generate_masks_with_special_tokens_and_transfer_map(tokenized, model.specical_tokens, model.tokenizer) # text_self_attention_masks: True for nomask, False for mask

    if text_self_attention_masks.shape[1] > model.max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, : model.max_text_len, : model.max_text_len]
        position_ids = position_ids[:, : model.max_text_len]
        tokenized['input_ids'] = tokenized['input_ids'][:, : model.max_text_len]
        tokenized['attention_mask'] = tokenized['attention_mask'][:, : model.max_text_len]
        tokenized['token_type_ids'] = tokenized['token_type_ids'][:, : model.max_text_len]

    # extract text embeddings
    if model.sub_sentence_present:
        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != 'attention_mask'}
        tokenized_for_encoder['attention_mask'] = text_self_attention_masks
        tokenized_for_encoder['position_ids'] = position_ids
    else:
        tokenized_for_encoder = tokenized
    bert_output = model.bert(**tokenized_for_encoder)  # bs, 195, 768
    encoded_text = model.feat_map(bert_output['last_hidden_state'])  # bs, 195, d_model
    text_token_mask = tokenized.attention_mask.bool()  # bs, 195: True for nomask, False for mask

    if encoded_text.shape[1] > model.max_text_len:
        encoded_text = encoded_text[:, : model.max_text_len, :]
        text_token_mask = text_token_mask[:, : model.max_text_len]
        position_ids = position_ids[:, : model.max_text_len]
        text_self_attention_masks = text_self_attention_masks[:, : model.max_text_len, : model.max_text_len]

    return {
        'caption':                   caption,
        'tokenized_input_ids':       tokenized['input_ids'],      # bs, L
        'tokenized_type_ids':        tokenized['token_type_ids'], # bs, L
        'encoded_text':              encoded_text,                # bs, L, d_model
        'text_token_mask':           text_token_mask,             # bs, L
        'position_ids':              position_ids,                # bs, L
        'text_self_attention_masks': text_self_attention_masks,   # bs, L, L
    }


def get_grounding_output_no_classify(model, text_dict, image, box_threshold):
    image = image.unsqueeze(0)
    if isinstance(image, (list, torch.Tensor)):
        image = nested_tensor_from_tensor_list(image)
    model.features, model.poss = model.backbone(image)

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
            m = image.mask
            mask = torch.nn.functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            srcs.append(src)
            masks.append(mask)
            model.poss.append(pos_l)

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
    logits = outputs_class[-1][0]  # (nq, 256)
    boxes = outputs_coord_list[-1][0]  # (nq, 4)

    # top k
    assert logits.size(0) == boxes.size(0) == 900, str(logits.size())
    logits_no_class = logits.max(dim=1)[0]
    logits_no_class, topk_idx = torch.topk(logits_no_class, 300)
    boxes = boxes[topk_idx]

    # # for intermediate outputs
    # if self.aux_loss:
    #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
    # # for encoder output
    # if hs_enc is not None:
    #     # prepare intermediate outputs
    #     interm_coord = ref_enc[-1]
    #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
    #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
    #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

    # filter output
    scores = logits_no_class.sigmoid()
    filt_mask = scores > box_threshold
    return boxes[filt_mask], scores[filt_mask]


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, images, image_transform, training=True):
        super(BoxDataset, self).__init__()
        self.images = images
        self.image_transform = image_transform
        self.training = training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im_i = self.images[i]
        im_torch, _ = self.image_transform(Image.open(im_i['file_name']).convert('RGB'), None)
        if not self.training:
            return im_torch, copy.deepcopy(im_i)

        targets = {'labels': [], 'boxes': []}
        for ann in im_i['annotations']:
            if ann['bbox_mode'] == 0:
                x1, y1, x2, y2 = ann['bbox']
                xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            else:
                assert ann['bbox_mode'] == 1
                x1, y1, w, h = ann['bbox']
                xc, yc = x1 + w / 2, y1 + h / 2
            xc, yc, w, h = xc / im_i['width'], yc / im_i['height'], w / im_i['width'], h / im_i['height']
            targets['labels'].append(0)
            targets['boxes'].append([xc, yc, w, h])

        targets['labels'] = torch.tensor(targets['labels']).long()
        if len(targets['boxes']): 
            targets['boxes'] = torch.tensor(targets['boxes']).float() 
        else: 
            targets['boxes'] = torch.empty((0, 4)).float()
        return im_torch, targets, im_i['video_id']

    @staticmethod
    def collate(batch):
        return list(zip(*batch))


def get_scenes100_images(args, keep_class=False):
    import hashlib
    video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
    video_id_list = sorted(video_id_list, key=lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
    video_id_list_train, video_id_list_test = map(sorted, [video_id_list[: 50], video_id_list[50 :]])
    print('training videos:', ' '.join(video_id_list_train))
    print('testing videos: ', ' '.join(video_id_list_test))

    images_train_fewshot, images_test_fewshot, images_eval = [], [], []
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        images = sorted(images, key=lambda x: x['file_name'])
        for im in images:
            im['video_id'] = video_id
            im['file_name'] = os.path.join(inputdir, 'unmasked', im['file_name'])
            if not keep_class:
                for ann in im['annotations']:
                    ann['category_id'] = 0   # make all classes become one.

        if video_id in video_id_list_train:
            images_train_fewshot.extend(images[: args.shot])
        else:
            images_test_fewshot.extend(images[: args.shot])
            images_eval.extend(images[args.shot :])
    print('scenes100 %d-shot: training/testing %d/%d, evaluation %d' % (args.shot, len(images_train_fewshot), len(images_test_fewshot), len(images_eval)))
    return images_train_fewshot, images_test_fewshot, images_eval


def evaluate_scenes100_masked_fewshot(images, detections):
    import skimage.io
    import imantics
    from evaluation import _check_overlap

    mask_rgb, mask_alpha = [0, 1, 0], 0.3
    images, detections = map(copy.deepcopy, [images, detections])
    for im1, im2 in zip(images, detections):
        assert im1['file_name'] == im2['file_name']
    im_arr_templates = {}
    for im in images:
        im_arr_templates[im['video_id']] = im['file_name']
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

    # convert BoxMode, remove boxes overlapping with mask
    for im in images:
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
    print('annotation: %d images %d bboxes' % (len(images), sum(map(lambda x: len(x['annotations']), images))))

    for im in detections:
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
    print('detections: %d images %d bboxes' % (len(detections), sum(map(lambda x: len(x['annotations']), detections))))
    return eval_AP(images, detections, return_thres=False)


def get_ovcoco_images(args):
    annotations_json = os.path.join(args.cocodir, 'annotations', 'ovd_ins_val2017_all.json')
   
    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)

    coco_dicts = {}
    images_dir = os.path.join(args.cocodir, 'images', 'val2017')
    for i, im in enumerate(annotations['images']):
        coco_dicts[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'image_id': im['id'], 'video_id': str(i), 'height': im['height'], 'width': im['width'], 'annotations': []}
    for ann in annotations['annotations']:
        x, y, w, h = ann['bbox']
        ann['bbox'] = [x, y, x + w, y + h]
        ann['bbox_mode'] = 0
        ann['category_id'] = 0

        coco_dicts[ann['image_id']]['annotations'].append(ann)
    coco_dicts = list(coco_dicts.values())
    coco_dicts = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts))
   
    for i in range(0, len(coco_dicts)):
        coco_dicts[i]['image_id'] = i + 1

    random.seed(42)
    random_indexes = random.sample(list(range(len(coco_dicts))), 65) # num classes 48 base + 17 novel
    images_fewshot, images_eval =  [], []
    for i in range(len(coco_dicts)):
        if i in random_indexes:
            images_fewshot.append(coco_dicts[i])
        else:
            images_eval.append(coco_dicts[i])
    # count_images, count_bboxes = len(coco_dicts), sum(map(lambda ann: len(ann['annotations']), coco_dicts))
    print('OVCOCO test set: %d training_images, %d eval_images' % (len(images_fewshot), len(images_eval)))
    return images_fewshot, images_eval


def get_rareplanes_images(args):
    def _convert(annotations_json, split='train'):
        with open(annotations_json, 'r') as fp:
            annotations = json.load(fp)

        coco_dicts = {}
        images_dir = os.path.join(args.rareplanes_dir, 'real', split, 'PS-RGB_tiled')
        for i, im in enumerate(annotations['images']):
            coco_dicts[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'image_id': im['id'], 'video_id': str(i), 'height': im['height'], 'width': im['width'], 'annotations': []}
        for ann in annotations['annotations']:
            x, y, w, h = ann['bbox']
            ann['bbox'] = [x, y, x + w, y + h]
            ann['bbox_mode'] = 0
            ann['category_id'] = 0

            coco_dicts[ann['image_id']]['annotations'].append(ann)
        coco_dicts = list(coco_dicts.values())
        coco_dicts = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts))
    
        for i in range(0, len(coco_dicts)):
            coco_dicts[i]['image_id'] = i + 1
        
        return coco_dicts
    
    train_annotations_json = os.path.join(args.rareplanes_dir, 'real', 'metadata_annotations', 'RarePlanes_Train_Coco_Annotations_tiled.json')
    test_annotations_json = os.path.join(args.rareplanes_dir, 'real', 'metadata_annotations', 'RarePlanes_Test_Coco_Annotations_tiled.json')
    
    train_dicts = _convert(train_annotations_json, split='train')
    test_dicts = _convert(test_annotations_json, split='test')

    random.seed(42)
    random_indexes = random.sample(list(range(len(train_dicts))), 50) # number of samples for few-shot learning
    images_fewshot, images_eval =  [], test_dicts
    for i in range(len(train_dicts)):
        if i in random_indexes:
            images_fewshot.append(train_dicts[i])

    # count_images, count_bboxes = len(coco_dicts), sum(map(lambda ann: len(ann['annotations']), coco_dicts))
    print('RarePlanes dataset: %d training_images, %d eval_images' % (len(images_fewshot), len(images_eval)))
    return images_fewshot, images_eval


def get_birdsai_images(args):
    from collections import defaultdict
    import cv2
    def _convert(data_path):
        images_per_video = {}
        folders = [f.name for f in os.scandir(os.path.join(data_path, 'images')) if f.is_dir()]

        for folder in folders:
            frames = sorted([os.path.join(os.path.join(data_path, 'images', folder), f) for f in os.listdir(os.path.join(data_path, 'images', folder)) if os.path.join(os.path.join(data_path, 'images', folder), f).endswith('.jpg')])
     
            sample = cv2.imread(frames[0])
            height, width, _ = sample.shape
            
            frames_dict = defaultdict(lambda: {'annotations': []})
            csv_filename = folder + '.csv'
            with open(os.path.join(data_path, 'annotations', csv_filename), mode='r') as file:
                reader = csv.reader(file)
                
                for row in reader:
                    frame_number = int(row[0])
                    x, y, w, h = map(int, row[2:6])
                    
                    # Convert bounding box from XYWH to XYXY format
                    bbox = [x, y, x + w, y + h]
                    
                    # Add the bbox to the corresponding frame's annotations
                    frames_dict[frame_number]['annotations'].append({'bbox': bbox, 'bbox_mode': 0, 'category_id': 0})

                    if 'video_id' not in frames_dict[frame_number]:
                        frames_dict[frame_number]['video_id'] = folder
                        frames_dict[frame_number]['file_name'] = frames[frame_number]
                        frames_dict[frame_number]['height'] = height
                        frames_dict[frame_number]['width'] = width

            images_per_video[folder] = [frames_dict[frame] for frame in sorted(frames_dict)]
        return images_per_video

    images_train_fewshot, images_test_fewshot, images_eval = [], [], []

    images_per_video = _convert(os.path.join(args.birdsai_dir, 'TrainReal'))
    print('training videos:', ' '.join(list(images_per_video.keys())))
    for video_id in images_per_video:
        images_train_fewshot.extend(images_per_video[video_id][: args.shot])

    images_per_video = _convert(os.path.join(args.birdsai_dir, 'TestReal'))
    print('testing videos:', ' '.join(list(images_per_video.keys())))
    for video_id in images_per_video:
        images_test_fewshot.extend(images_per_video[video_id][: args.shot])
        images_eval.extend(images_per_video[video_id][args.shot:])

    print('BIRDSAI %d-shot: training/testing %d/%d, evaluation %d' % (args.shot, len(images_train_fewshot), len(images_test_fewshot), len(images_eval)))
    return images_train_fewshot, images_test_fewshot, images_eval


def get_egoper_images(args):
    def _convert(dataset):
        images = {im['id']: im for im in filter(lambda x: x['file_name'].find('coco') < 0, dataset['images'])}
        for im in images.values():
            im['video_id'] = im['video']
            im['file_name'] = im['file_name'].replace('\\', '/')
            im['file_name'] = os.path.join(args.egoper_dir, im['file_name'])
            del im['license'], im['flickr_url'], im['coco_url'], im['date_captured'], im['video']
            im['annotations'] = []
        for ann in dataset['annotations']:
            if ann['image_id'] in images:
                x, y, w, h = ann['bbox']
                ann['bbox'] = [x, y, x + w, y + h]
                ann['bbox_mode'] = 0
                ann['category_id'] = 0
                images[ann['image_id']]['annotations'].append(ann)
                del ann['attributes'], ann['id'], ann['image_id']
        images_per_video = {}
        for im in images.values():
            if not im['video_id'] in images_per_video:
                images_per_video[im['video_id']] = []
            images_per_video[im['video_id']].append(im)
        images_per_video = {k: v for k, v in images_per_video.items() if len(v) >= 5}
        for video_id in images_per_video:
            images_per_video[video_id] = sorted(images_per_video[video_id], key=lambda x: x['file_name'])
        return images_per_video

    images_train_fewshot, images_test_fewshot, images_eval = [], [], []
    with open(os.path.join(args.egoper_dir, 'data', 'train_5_recipes.json'), 'r') as fp:
        images_per_video = _convert(json.load(fp))
    print('training videos:', ' '.join(list(images_per_video.keys())))
    for video_id in images_per_video:
        images_train_fewshot.extend(images_per_video[video_id][: args.shot])
    with open(os.path.join(args.egoper_dir, 'data', 'valid_5_recipes.json'), 'r') as fp:
        images_per_video = _convert(json.load(fp))
    print('testing videos: ', ' '.join(list(images_per_video.keys())))
    for video_id in images_per_video:
        images_test_fewshot.extend(images_per_video[video_id][: args.shot])
        images_eval.extend(images_per_video[video_id][args.shot :])

    print('EgoPER %d-shot: training/testing %d/%d, evaluation %d' % (args.shot, len(images_train_fewshot), len(images_test_fewshot), len(images_eval)))
    return images_train_fewshot, images_test_fewshot, images_eval


def get_hoist_images(args):
    import hashlib
    def _convert(dataset, split):
        videos = {v['id']: v['name'] for v in dataset['videos']}
        images = {im['id']: im for im in dataset['images']}
        for im in images.values():
            im['video_id'] = videos[im['video_id']]
            im['file_name'] = os.path.join(args.hoist_dir, split, 'JPEGImages', im['file_name'])
            im['annotations'] = []
        for ann in dataset['annotations']:
            if ann['image_id'] in images:
                x, y, w, h = ann['bbox']
                if w < 1 or h < 1:
                    continue
                ann['bbox'] = [x, y, x + w, y + h]
                ann['bbox_mode'] = 0
                ann['category_id'] = 0
                ann['segmentation'] = []
                images[ann['image_id']]['annotations'].append(ann)
                del ann['instance_id'], ann['id'], ann['image_id']
        images_per_video = {}
        for im in images.values():
            if not im['video_id'] in images_per_video:
                images_per_video[im['video_id']] = []
            images_per_video[im['video_id']].append(im)
        images_per_video = {k: v for k, v in images_per_video.items() if len(v) >= 5}
        for video_id in images_per_video:
            images_per_video[video_id] = sorted(images_per_video[video_id], key=lambda x: x['file_name'])
        return images_per_video

    with open(os.path.join(args.hoist_dir, 'annotations', 'train.json'), 'r') as fp:
        images_per_video = _convert(json.load(fp), 'train')
    video_id_list = sorted(images_per_video.keys(), key=lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
    video_id_list_train, video_id_list_test = map(sorted, [video_id_list[: len(video_id_list) // 2], video_id_list[len(video_id_list) // 2 :]])
    print('training videos:', ' '.join(video_id_list_train[:10]), '...', ' '.join(video_id_list_train[-10:]))
    print('testing videos: ', ' '.join(video_id_list_test[:10]), '...', ' '.join(video_id_list_test[-10:]))

    images_train_fewshot, images_test_fewshot, images_eval = [], [], []
    for video_id in video_id_list:
        if video_id in video_id_list_train:
            images_train_fewshot.extend(images_per_video[video_id][: args.shot])
        else:
            images_test_fewshot.extend(images_per_video[video_id][: args.shot])
            images_eval.extend(images_per_video[video_id][args.shot :])
    print('HOIST %d-shot: training/testing %d/%d, evaluation %d' % (args.shot, len(images_train_fewshot), len(images_test_fewshot), len(images_eval)))
    return images_train_fewshot, images_test_fewshot, images_eval


@torch.no_grad()
def detect(args):
    image_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
    with torch.no_grad():
        text_dict = get_text_dict(model, args.prompt)
    torch.cuda.empty_cache()
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
        with torch.no_grad():
            boxes, scores = get_grounding_output_no_classify(model, copy.deepcopy(text_dict), im_torch.cuda(), args.box_threshold)
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
def detect_keep_class(args):
    image_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
    with torch.no_grad():
        text_dict = get_text_dict(model, args.prompt)
    torch.cuda.empty_cache()
    if args.dataset == 'scenes100':
        _, _, images_eval = get_scenes100_images(args, keep_class=True)
    if args.dataset == 'egoper':
        _, _, images_eval = get_egoper_images(args)
    if args.dataset == 'hoist':
        _, _, images_eval = get_hoist_images(args)

    loader = torch.utils.data.DataLoader(
        BoxDataset(images_eval, image_transform, training=False),
        batch_size=None, shuffle=False, num_workers=4
    )

    detections = []
    for im_torch, im in tqdm.tqdm(loader, ascii=True, total=len(images_eval)):
        im['annotations'] = []
        with torch.no_grad():
            boxes, scores = get_grounding_output_no_classify(model, copy.deepcopy(text_dict), im_torch.cuda(), args.box_threshold)
        for box, s in zip(boxes, scores):
            xc, yc, w, h = map(float, [box[0] * im['width'], box[1] * im['height'], box[2] * im['width'], box[3] * im['height']])
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': args.inclusive_class, 'score': float(s), 'bbox_mode': 0})
        detections.append(im)

    if args.dataset == 'scenes100':
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs = evaluate_scenes100_masked_fewshot(images_eval, detections)
    if args.dataset == 'egoper' or args.dataset == 'hoist':
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs = eval_AP(images_eval, detections, return_thres=False)
    print(APs['results'])


def detect_finegrain_prompt_egoper(args):
    with open(os.path.join(args.egoper_dir, 'data', 'train_5_recipes.json'), 'r') as fp:
        dataset = json.load(fp)
    names = [c['name'] for c in dataset['categories'][:40]]
    prompt = ' . '.join(names) + ' .'
    print(prompt)
    args.prompt = prompt
    args.dataset = 'egoper'
    detect(args)


@torch.no_grad()
def detect_finegrain_annotate_hoist(args):
    import skimage.io
    import csv
    clip_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'CLIP')
    sys.path.append(clip_dir)
    import clip

    clip_model, preprocess = clip.load('ViT-B/32', device='cuda')

    with open(os.path.join(clip_dir, 'oidv6-class-descriptions.csv'), 'r') as fp:
        reader = csv.reader(fp)
        candidates_name = [row[1].lower() for row in reader][1:]
    candidates_desc = ['a photo of %s' % c for c in candidates_name]
    candidates_tokenized = clip.tokenize(candidates_desc).to('cuda')
    candidates_features = clip_model.encode_text(candidates_tokenized)
    candidates_features = candidates_features / candidates_features.norm(dim=1, keepdim=True)
    del candidates_tokenized
    torch.cuda.empty_cache()
    print(candidates_name[:10], len(candidates_name))
    print(candidates_desc[:10], len(candidates_desc))
    print(candidates_features.size())

    _, images_test_fewshot, _ = get_hoist_images(args)
    images_test_fewshot = images_test_fewshot
    names_per_video = {}
    for im in tqdm.tqdm(images_test_fewshot, ascii=True, desc='cropping'):
        if not im['video_id'] in names_per_video:
            names_per_video[im['video_id']] = []
        im_arr = skimage.io.imread(im['file_name'])
        for ann in im['annotations']:
            assert ann['bbox_mode'] == 0
            x1, y1, x2, y2 = map(int, ann['bbox'])
            names_per_video[im['video_id']].append(im_arr[y1 : y2, x1 : x2, :])

    for v in tqdm.tqdm(names_per_video, ascii=True, desc='classifying'):
        if len(names_per_video[v]) > 0:
            crops = list(map(lambda x: preprocess(Image.fromarray(x)).cuda(), names_per_video[v]))
            crops = torch.stack(crops, dim=0)
            crops_feature = clip_model.encode_image(crops)
            crops_feature = crops_feature / crops_feature.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * crops_feature @ candidates_features.t()
            candidates_idx = list(map(int, torch.argmax(logits_per_image, dim=1).detach().cpu().numpy()))
            names_per_video[v] = candidates_idx

            # plt.figure()
            # for i, (im_arr, n) in enumerate(zip(names_per_video[v], names)):
            #     plt.subplot(1, len(names), i + 1)
            #     plt.imshow(im_arr)
            #     plt.title(n)
            # plt.suptitle('%s %s %s' % (v, candidates_idx, names))
            # plt.show()
    filename = os.path.join(os.path.dirname(__file__), 'hoist_eval_prompts_clip_openimages.json')
    with open(filename, 'w') as fp:
        json.dump({'idx': names_per_video, 'candidates_name': candidates_name}, fp)


@torch.no_grad()
def detect_finegrain_detect_hoist(args):
    image_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()

    _, _, images_eval = get_hoist_images(args)
    video_id_list = set([im['video_id'] for im in images_eval])

    with open(os.path.join(os.path.dirname(__file__), 'hoist_eval_prompts_clip_openimages.json'), 'r') as fp:
        meta = json.load(fp)
    candidates_name = np.array(meta['candidates_name'])
    name_idx_per_video = meta['idx']

    prompts = {}
    for v in video_id_list:
        if len(name_idx_per_video[v]) < 1:
            prompts[v] = args.prompt
        else:
            _names = candidates_name[name_idx_per_video[v]]
            _names = [x + ' in hand . ' for x in _names]
            _names = ''.join(_names) + args.prompt
            prompts[v] = _names

    text_dict_per_video = {}
    for v in tqdm.tqdm(video_id_list, ascii=True):
        text_dict_per_video[v] = get_text_dict(model, prompts[v])
    loader = torch.utils.data.DataLoader(
        BoxDataset(images_eval, image_transform, training=False),
        batch_size=None, shuffle=False, num_workers=4
    )
    detections = []
    for im_torch, im in tqdm.tqdm(loader, ascii=True, total=len(images_eval)):
        im['annotations'] = []
        with torch.no_grad():
            boxes, scores = get_grounding_output_no_classify(model, copy.deepcopy(text_dict_per_video[im['video_id']]), im_torch.cuda(), args.box_threshold)
        for box, s in zip(boxes, scores):
            xc, yc, w, h = map(float, [box[0] * im['width'], box[1] * im['height'], box[2] * im['width'], box[3] * im['height']])
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': 0, 'score': float(s), 'bbox_mode': 0})
        detections.append(im)
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

    image_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
    with torch.no_grad():
        text_dict = get_text_dict(model, args.prompt)
    torch.cuda.empty_cache()
    if args.dataset == 'ovcoco':
        _, images_eval = get_ovcoco_images(args)
    if args.dataset == 'rareplanes':
        _, images_eval = get_rareplanes_images(args)
    if args.dataset == 'scenes100':
        _, _, images_eval = get_scenes100_images(args)
    if args.dataset == 'egoper':
        _, _, images_eval = get_egoper_images(args)
    if args.dataset == 'birdsai':
        _, _, images_eval = get_birdsai_images(args)
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
        boxes, scores = get_grounding_output_no_classify(model, copy.deepcopy(text_dict), im_torch.cuda(), args.box_threshold)
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
        gt = _images[-1]['annotations']
        pred = list(filter(lambda x: x['score'] > t, _detections[-1]['annotations']))
        skimage.io.imsave(os.path.join(outputdir, '%s_%s_gt.jpg' % (args.dataset, video_id)), draw_bbox(im, gt, 'Ground Truth', font, math.ceil(_WH / 500)), quality=90)
        skimage.io.imsave(os.path.join(outputdir, '%s_%s_unadapt.jpg' % (args.dataset, video_id)), draw_bbox(im, pred, 'Not Adapted', font, math.ceil(_WH / 500)), quality=90)


@torch.no_grad()
def inference_throughput(args):
    import gc
    import time

    image_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
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
            ret = get_grounding_output_no_classify(model, text_dict, im_torch, args.box_threshold)
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Grounding DINO example', add_help=True)
    parser.add_argument('--opt', type=str, default='detect')
    parser.add_argument('--config', '-c', type=str, default='swint', choices=['swint', 'swinb'], help='path to config file')
    # parser.add_argument('--checkpoint_path', '-p', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--box_threshold', type=float, default=0.05, help='box threshold')
    # parser.add_argument('--text_threshold', type=float, default=0.25, help='text threshold')
    parser.add_argument('--inclusive_class', type=int, default=0, choices=[0, 1])

    parser.add_argument('--dataset', type=str, default=None, choices=['scenes100', 'egoper', 'hoist', 'ovcoco', 'birdsai', 'rareplanes'])
    parser.add_argument('--cocodir', type=str, default='../../MSCOCO2017')
    parser.add_argument('--birdsai_dir', type=str, default='../../../BIRDSAI')
    parser.add_argument('--egoper_dir', type=str, default='../../../PTG/ptg_detection')
    parser.add_argument('--hoist_dir', type=str, default='../../../OIH_VIS')
    parser.add_argument('--rareplanes_dir', type=str, default='../../../RarePlanes')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--shot', type=int, default=1, help='few-shot learning')
    args = parser.parse_args()
    print(args)

    if args.opt == 'detect':
        detect(args)
    if args.opt == 'detect_kc':
        detect_keep_class(args)
    if args.opt == 'case_study':
        show_detected(args)
    if args.opt == 'finegrain_ego':
        detect_finegrain_prompt_egoper(args)
    if args.opt == 'finegrain_hoist':
        detect_finegrain_annotate_hoist(args)
    if args.opt == 'finegrain_hoist_detect':
        detect_finegrain_detect_hoist(args)
    if args.opt == 'tp':
        inference_throughput(args)

'''
python inference_groundingdino_accv.py --dataset scenes100 --shot 1 --prompt "person . vehicle ."
python inference_groundingdino_accv.py --dataset scenes100 --shot 1 --prompt "person . pedestrian . vehicle . automobile . car ."

python inference_groundingdino_accv.py --dataset scenes100 --shot 1 --prompt "person ." --opt detect_kc --inclusive_class 0

python inference_groundingdino_accv.py --dataset egoper --shot 1 --prompt "kitchen . cooking . human body ."
python inference_groundingdino_accv.py --dataset egoper --shot 1 --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ."

python inference_groundingdino_accv.py --dataset hoist --shot 1 --prompt "hand-held object ."
python inference_groundingdino_accv.py --dataset hoist --shot 1 --prompt "object in hand ."

python inference_groundingdino_accv.py --dataset ovcoco --shot 1 --prompt "coco objects ."

python inference_groundingdino_accv.py --dataset birdsai --shot 1 --prompt "nighttime images of animals and humans ."

python inference_groundingdino_accv.py --dataset rareplanes --shot 1 --prompt "planes ."

python inference_groundingdino_accv.py --opt finegrain_ego --shot 1
python inference_groundingdino_accv.py --dataset egoper --shot 1 --prompt "kettle . measuring cup . kitchen scale . coffee grinder . filter cone dripper . paper basket filter . mug . thermometer . spoon . bowl . bottle . hand . paper filter . tortilla . peanut butter . jam . toothpick . knife . chopping board . dental floss . paper towel . plate . paper sheet . tea bag . trash can . honey . nutella . banana slice . cinnamon . oat . microwave . raisin ."

python inference_groundingdino_accv.py --opt finegrain_hoist --shot 1
python inference_groundingdino_accv.py --opt finegrain_hoist_detect --shot 1 --prompt "hand-held object ."

python inference_groundingdino_accv.py --opt case_study --shot 1 --dataset scenes100 --prompt "person . vehicle ."
python inference_groundingdino_accv.py --opt case_study --shot 1 --dataset egoper --prompt "kitchen object ."
python inference_groundingdino_accv.py --opt case_study --shot 1 --dataset hoist --prompt "hand-held object ."

python inference_groundingdino_accv.py --opt tp --shot 1 --dataset egoper --prompt "kitchen object ."
python inference_groundingdino_accv.py --opt tp --shot 1 --dataset egoper --prompt "kitchen . cooking . human body ."
python inference_groundingdino_accv.py --opt tp --shot 1 --dataset egoper --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ."
python inference_groundingdino_accv.py --opt tp --shot 1 --dataset egoper --prompt "kettle . measuring cup . kitchen scale . coffee grinder . filter cone dripper . paper basket filter . mug . thermometer . spoon . bowl . bottle . hand . paper filter . tortilla . peanut butter . jam . toothpick . knife . chopping board . dental floss . paper towel . plate . paper sheet . tea bag . trash can . honey . nutella . banana slice . cinnamon . oat . microwave . raisin ."
'''

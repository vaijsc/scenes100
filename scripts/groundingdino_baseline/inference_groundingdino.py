import argparse
import os
import sys
import json
import tqdm
import copy
import contextlib

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch
from detectron2.structures import BoxMode

groundingdino_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GroundingDINO')
sys.path.append(groundingdino_dir)
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation import evaluate_masked

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # ERROR: huggingface/tokenizers: The current process just got forked, after parallelism has already been used.

video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
bbox_rgbs = ['#FF0000', '#0000FF']


def load_model(config):
    if config == 'swint':
        config_file = os.path.join(groundingdino_dir, 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
        checkpoint_path = os.path.join(groundingdino_dir, 'groundingdino_swint_ogc.pth')
    else:
        assert config == 'swinb'
        config_file = os.path.join(groundingdino_dir, 'groundingdino', 'config', 'GroundingDINO_SwinB_cfg.py')
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


def count_parameters(args):
    def _count(m):
        N = 0
        for p in m.parameters():
            N += p.numel()
        return N

    model = load_model(args.config)
    print('total:      %11d' % _count(model))
    print('bert:       %11d' % _count(model.bert))
    print('input_proj: %11d' % _count(model.input_proj))
    print('backbone:   %11d' % _count(model.backbone))
    print('feat_map:   %11d' % _count(model.feat_map))
    print('transformer:%11d' % _count(model.transformer))
    print('bbox_embed: %11d' % _count(model.bbox_embed))
    print('class_embed:%11d' % _count(model.class_embed))


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


def get_text_dict(model, prompt_per_class):
    assert len(prompt_per_class) == 2
    # text_prompt_0 = 'person . people . pedestrian . driver .' # 37.58
    # text_prompt_1 = 'vehicle . car . bus . truck .'           # 65.13
    # text_prompt_0 = 'person people pedestrian driver .' # 20.49
    # text_prompt_1 = 'vehicle car bus truck .'           # 55.35
    # text_prompt_0 = 'person .'  # 43.91
    # text_prompt_1 = 'vehicle .' # 44.42
    # text_prompt_0 = 'people .'     # 39.57
    # text_prompt_1 = 'automobile .' # 63.16
    print('prompts:\n%s\n%s' % tuple(prompt_per_class))
    text_dict_list = []
    for caption in prompt_per_class:
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

        text_dict_list.append({
            'caption':                   caption,
            'tokenized_input_ids':       tokenized['input_ids'],      # bs, L
            'tokenized_type_ids':        tokenized['token_type_ids'], # bs, L
            'encoded_text':              encoded_text,                # bs, L, d_model
            'text_token_mask':           text_token_mask,             # bs, L
            'position_ids':              position_ids,                # bs, L
            'text_self_attention_masks': text_self_attention_masks,   # bs, L, L
        })
        # print(caption, encoded_text.size(), text_token_mask, position_ids, text_self_attention_masks)
    return text_dict_list


def get_grounding_output_no_classify(model, text_dict_list, image, box_threshold):
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

    boxes_per_class, scores_per_class = [], []
    for text_dict in text_dict_list:
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
        boxes_per_class.append(boxes[filt_mask])
        scores_per_class.append(scores[filt_mask])
    return boxes_per_class, scores_per_class


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

        targets_per_class = [
            {'labels': [], 'boxes': []},
            {'labels': [], 'boxes': []}
        ]
        for ann in im_i['annotations']:
            if ann['bbox_mode'] == BoxMode.XYXY_ABS:
                x1, y1, x2, y2 = ann['bbox']
                xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            else:
                assert ann['bbox_mode'] == BoxMode.XYWH_ABS
                x1, y1, w, h = ann['bbox']
                xc, yc = x1 + w / 2, y1 + h / 2
            xc, yc, w, h = xc / im_i['width'], yc / im_i['height'], w / im_i['width'], h / im_i['height']
            targets_per_class[ann['category_id']]['labels'].append(0)
            targets_per_class[ann['category_id']]['boxes'].append([xc, yc, w, h])

        for c in range(0, len(targets_per_class)):
            targets_per_class[c]['labels'] = torch.tensor(targets_per_class[c]['labels']).long()
            targets_per_class[c]['boxes'] = torch.tensor(targets_per_class[c]['boxes']).float()
        return im_torch, targets_per_class, im_i['video_id']

    @staticmethod
    def collate(batch):
        return list(zip(*batch))


def detect_scenes100(args):
    image_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
    with torch.no_grad():
        text_dict_list = get_text_dict(model, args.prompts)
    torch.cuda.empty_cache()

    images_all_video = []
    for video_id in (video_id_list if args.id == 'batch' else [args.id]):
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
        with torch.no_grad():
            boxes_per_class, scores_per_class = get_grounding_output_no_classify(model, copy.deepcopy(text_dict_list), im_torch.cuda(), args.box_threshold)
        for c in range(0, len(boxes_per_class)):
            for box, s in zip(boxes_per_class[c], scores_per_class[c]):
                xc, yc, w, h = map(float, [box[0] * im['width'], box[1] * im['height'], box[2] * im['width'], box[3] * im['height']])
                x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': c, 'score': float(s), 'bbox_mode': BoxMode.XYXY_ABS})
        detections[im['video_id']].append(im)

    APs_all = {}
    for video_id in detections:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs_all[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)
        del APs_all[video_id]['raw']
        print(video_id, APs_all[video_id]['results'])
    categories = ['person', 'vehicle', 'overall', 'weighted']
    print('videos average:')
    for c in categories:
        _AP_videos = np.array([APs_all[v]['results'][c] for v in APs_all]) * 100
        print(c, _AP_videos[_AP_videos[:, 0] >= 0].mean(axis=0))


def case_study(args):
    assert args.id != 'batch'
    image_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
    with torch.no_grad():
        text_dict_list = get_text_dict(model, args.prompts)
    torch.cuda.empty_cache()

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    for im in images:
        im['file_name'] = os.path.join(inputdir, 'unmasked', im['file_name'])
    dst = BoxDataset(images, image_transform, training=False)

    detections = []
    for im_torch, im in tqdm.tqdm(dst, ascii=True):
        im['file_name'] = os.path.basename(im['file_name'])
        im['annotations'] = []
        with torch.no_grad():
            boxes_per_class, scores_per_class = get_grounding_output_no_classify(model, copy.deepcopy(text_dict_list), im_torch.cuda(), args.box_threshold)
        for c in range(0, len(boxes_per_class)):
            for box, s in zip(boxes_per_class[c], scores_per_class[c]):
                xc, yc, w, h = map(float, [box[0] * im['width'], box[1] * im['height'], box[2] * im['width'], box[3] * im['height']])
                x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': c, 'score': float(s), 'bbox_mode': BoxMode.XYXY_ABS})
        detections.append(im)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        APs = evaluate_masked(args.id, detections, outputfile=None)
    print(args.id, APs['results'])

    import matplotlib.patches as patches
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    im_arr = Image.open(images[0]['file_name']).convert('RGB')
    ax1.imshow(im_arr)
    for ann in images[0]['annotations']:
        x1, y1, x2, y2 = ann['bbox']
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bbox_rgbs[ann['category_id']], facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title('%s %s\nGround Truth' % (args.id, os.path.basename(images[0]['file_name'])))
    ax1.set_axis_off()
    ax2.imshow(im_arr)
    for ann in detections[0]['annotations']:
        # if ann['score'] < 0.1:
        #     continue
        x1, y1, x2, y2 = ann['bbox']
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bbox_rgbs[ann['category_id']], facecolor='none')
        ax2.add_patch(rect)
    ax2.set_title('class-0 \"%s\" $AP^m$=%.2f\nclass-1 \"%s\" $AP^m$=%.2f' % (args.prompts[0], APs['results']['person'][0] * 100, args.prompts[1], APs['results']['vehicle'][0] * 100))
    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Grounding DINO example', add_help=True)
    parser.add_argument('--opt', type=str)
    parser.add_argument('--config', '-c', type=str, default='swint', choices=['swint', 'swinb'], help='path to config file')
    # parser.add_argument('--checkpoint_path', '-p', type=str, required=True, help='path to checkpoint file')
    # parser.add_argument('--output_dir', '-o', type=str, default='outputs', required=True, help='output directory')
    parser.add_argument('--box_threshold', type=float, default=0.05, help='box threshold')
    # parser.add_argument('--text_threshold', type=float, default=0.25, help='text threshold')
    parser.add_argument('--id', type=str, default='batch')
    parser.add_argument('--prompts', type=str, nargs='+', default=['person .', 'vehicle . car . bus . truck .'])
    args = parser.parse_args()
    print(args)

    if args.opt == 'detect':
        detect_scenes100(args)
    if args.opt == 'count':
        count_parameters(args)
    if args.opt == 'case':
        case_study(args)

'''
python inference_groundingdino.py --opt detect --config swint --id 001 --prompts "people ." "car ."
001 {'overall': [0.7095907716898655, 0.9160773366044233], 'person': [0.5637709386665543, 0.879890864511734], 'vehicle': [0.8554106047131769, 0.9522638086971128], 'weighted': [0.6545489192247282, 0.9024182175609857]}

python inference_groundingdino.py --opt detect --config swint --id 001 --prompts "person ." "automobile ."
001 {'overall': [0.6949533378884912, 0.8790396266142367], 'person': [0.5408070673631792, 0.7862472604143915], 'vehicle': [0.8490996084138033, 0.971831992814082], 'weighted': [0.6367685484507678, 0.8440137757106332]}

python inference_groundingdino.py --opt detect --config swint --id 001 --prompts "pedestrian ." "vehicle ."
001 {'overall': [0.6099218121728898, 0.8037003602321279], 'person': [0.5709730979014946, 0.8709419365871182], 'vehicle': [0.6488705264442851, 0.7364587838771377], 'weighted': [0.5952200439690393, 0.8290816876449975]}
'''

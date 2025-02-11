import argparse
import os
import sys
import gc
import json
import tqdm
import copy
import contextlib
import re

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T_tv
from detectron2.structures import BoxMode

# import yolo_world
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation import eval_AP
from dino.matcher import HungarianMatcher
from dino.criterion import SetCriterion

from inference_yolow_accv import load_model, BoxDataset, get_scenes100_images, evaluate_scenes100_masked_fewshot, get_egoper_images, get_hoist_images

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # ERROR: huggingface/tokenizers: The current process just got forked, after parallelism has already been used.


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
bbox_rgbs = ['#FF0000', '#0000FF']

def create_padding_mask(image_shape, padding):
    H, W = image_shape
    pad_top, pad_bottom, pad_left, pad_right = padding
    # breakpoint()
    # Initialize mask of zeros
    mask = torch.zeros((H, W), dtype=torch.bool)
    # breakpoint()
    if pad_top > 0:
        # Set top padding to 1
        mask[:int(pad_top), :] = True
    if pad_bottom > 0:
        # Set bottom padding to 1
        mask[-int(pad_bottom):, :] = True
    if pad_left > 0:
        # Set left padding to 1
        mask[:, :int(pad_left)] = True
    if pad_right > 0:
        # Set right padding to 1
        mask[:, -int(pad_right):] = True
    # breakpoint()
    return mask


class PromptEnhancer(torch.nn.Module):
    def __init__(self):
        super(PromptEnhancer, self).__init__()
        self.decoder = torch.nn.TransformerDecoderLayer(512, 8, batch_first=True)
        self.input_proj = [torch.nn.Sequential(torch.nn.Conv2d(384, 512, kernel_size=1), torch.nn.GroupNorm(32, 512)).cuda(),
                            torch.nn.Sequential(torch.nn.Conv2d(768, 512, kernel_size=1), torch.nn.GroupNorm(32, 512)).cuda(),
                            torch.nn.Sequential(torch.nn.Conv2d(768, 512, kernel_size=1), torch.nn.GroupNorm(32, 512)).cuda()]

    def forward(self, text_embeds, im_embeds, img_masks):
        im_embeds_ = copy.deepcopy(im_embeds)
        im_embeds_ = list(im_embeds_)
        for i, im_embed in enumerate(im_embeds_):
            im_embeds_[i] = self.input_proj[i](im_embed)
        
        im_embeds_ = [s.view(s.shape[0], s.shape[1], -1).transpose(1, 2) for s in im_embeds_]
        im_embeds_ = torch.cat(im_embeds_, dim=1) 
        
        return self.decoder(
            text_embeds,
            im_embeds_,
            memory_key_padding_mask=img_masks
        )


def model_forward(model, batch_inputs, batch_data_samples, arch, with_nms=True, keep_logits=True):
    data_batch = dict(inputs=batch_inputs,
                      data_samples=batch_data_samples)
    data_batch = model.data_preprocessor(data_batch, training=False)
    img_feats, txt_feats = model.extract_feat(data_batch['inputs'],
                                                data_batch['data_samples'])

    mask = []
    for data_sample in batch_data_samples:
        mask.append(create_padding_mask(batch_inputs.shape[-2:], data_sample.metainfo['pad_param']))
    
    mask = torch.stack(mask)

    img_masks = []
    for img_feat in img_feats:
        img_masks.append(torch.nn.functional.interpolate(mask[None].float(), size=img_feat.shape[-2:]).to(torch.bool)[0])

    img_masks = [_m.view(_m.size(0), -1).detach() for _m in img_masks]
    img_masks = torch.cat(img_masks, dim=1).cuda()
    # breakpoint()                                 
    if arch == 'enhancer':
        txt_feats = model.enhancer(txt_feats, img_feats, img_masks)
                                                
    model.bbox_head.num_classes = txt_feats[0].shape[0]

    if keep_logits:
        results_list = model.bbox_head.predict_train(img_feats,
                                            txt_feats,
                                            data_batch['data_samples'],
                                            rescale=True,
                                            with_nms=with_nms)
    else:
        results_list = model.bbox_head.predict(img_feats,
                                            txt_feats,
                                            data_batch['data_samples'],
                                            rescale=True,
                                            with_nms=with_nms)
 
    return results_list

def train_few_shot(args):
    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.join('./work_dirs')

    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline_cfg[1].type = 'mmyolo.YOLOv5KeepRatioResize'
    test_pipeline_cfg[2].type = 'mmyolo.LetterResize'
    test_pipeline_cfg[4].type = 'mmyolo.LoadText'
    
    test_pipeline = Compose(test_pipeline_cfg)
    
    print('box threshold: %f' % args.box_threshold)
    prompt = [category.strip() for category in re.split(r'\s*\.\s*', args.prompt) if category] + [' ']
    prompt = [[p] for p in prompt]

    if args.arch == 'enhancer':
        model = load_model(args).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        model.enhancer = PromptEnhancer().cuda()
        if args.ckpt is not None:
            print('loading from', args.ckpt)
            model.enhancer.load_state_dict(torch.load(args.ckpt))
        model.enhancer.train()
        optimizer = torch.optim.Adam(model.enhancer.parameters(), lr=args.lr)

        model.reparameterize(prompt)

    elif args.arch == 'head':
        model = load_model(args).cuda()
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

        # with torch.no_grad():
        #     text_dict_cache = get_text_dict(model, args.prompt)


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

    if args.dataset == 'scenes100':
        images_train_fewshot, images_test_fewshot, _ = get_scenes100_images(args)
    if args.dataset == 'egoper':
        images_train_fewshot, images_test_fewshot, _ = get_egoper_images(args)
    if args.dataset == 'hoist':
        images_train_fewshot, images_test_fewshot, _ = get_hoist_images(args)
    images_fewshot = (images_train_fewshot + images_test_fewshot) if args.setting == 'seen' else images_train_fewshot
    print('%s %d-shot %s training set: %d images' % (args.dataset, args.shot, args.setting, len(images_fewshot)))
    prefix = os.path.join(args.savedir, '%s_%s_%dshot_%s_%s' % (os.path.basename(args.config)[:-3], args.dataset, args.shot, args.setting, args.arch))
    print(prefix)

    dst_train = BoxDataset(images_fewshot, prompt, test_pipeline)
    loader_train = torch.utils.data.DataLoader(dst_train, batch_size=args.image_batch_size, collate_fn=BoxDataset.collate, shuffle=True, num_workers=args.num_workers)
    iter_train = iter(loader_train)

    eval_iters = list(range(0, args.iters, args.eval_interval)) + [args.iters - 1]
    for it in tqdm.tqdm(range(0, args.iters), ascii=True):
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(loader_train)
            batch = next(iter_train)
        batch_inputs, batch_data_samples, targets = batch
        optimizer.zero_grad()

        loss_dict = {'loss_ce': 0, 'loss_bbox': 0, 'loss_giou': 0}
        results = model_forward(model, batch_inputs.cuda(), batch_data_samples, args.arch, with_nms=False)
        
        targets_all, outputs_all = [], {'pred_boxes': [], 'pred_logits': []}
        for i, result in enumerate(results):
            target = targets[i]
            target['boxes'], target['labels'] = target['boxes'].cuda(), target['labels'].cuda()
            targets_all.append(target)
            
            H, W = batch_data_samples[i].metainfo['ori_shape']
            for i in range(result.bboxes.shape[0]):
                result.bboxes[i][0] = (result.bboxes[i][0] + result.bboxes[i][2]) / 2
                result.bboxes[i][1] = (result.bboxes[i][1] + result.bboxes[i][3]) / 2
                result.bboxes[i][2] = 2 * (result.bboxes[i][2] - result.bboxes[i][0])
                result.bboxes[i][3] = 2 * (result.bboxes[i][3] - result.bboxes[i][1])

                result.bboxes[i][0] /= W
                result.bboxes[i][1] /= H
                result.bboxes[i][2] /= W
                result.bboxes[i][3] /= H
            # breakpoint()
            outputs_all['pred_boxes'].append(result.bboxes)
            outputs_all['pred_logits'].append(result.scores)

        outputs_all['pred_boxes'] = torch.stack(outputs_all['pred_boxes'], dim=0)
        outputs_all['pred_logits'] = torch.stack(outputs_all['pred_logits'], dim=0).unsqueeze(2)
        loss_dict = loss_fn(outputs_all, targets_all)
    
        L = loss_dict['loss_ce'] * args.cls_loss_coef + loss_dict['loss_bbox'] * args.bbox_loss_coef + loss_dict['loss_giou'] * args.giou_loss_coef
        L.backward()
        print("loss:", L.item())
        optimizer.step()

        loss_history_dict['loss_ce'].append(float(loss_dict['loss_ce'].item()))
        loss_history_dict['loss_bbox'].append(float(loss_dict['loss_bbox'].item()))
        loss_history_dict['loss_giou'].append(float(loss_dict['loss_giou'].item()))
        loss_history_dict['weighted'].append(float(L.item()))

        if it in eval_iters:
            if args.arch == 'enhancer':
                torch.save(model.enhancer.state_dict(), prefix + f'_iter_{it}.pth')
            elif args.arch == 'head':
                torch.save(model.transformer.state_dict(), prefix + '.pth')
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
    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.join('./work_dirs')

    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline_cfg[1].type = 'mmyolo.YOLOv5KeepRatioResize'
    test_pipeline_cfg[2].type = 'mmyolo.LetterResize'
    test_pipeline_cfg[4].type = 'mmyolo.LoadText'
    
    test_pipeline = Compose(test_pipeline_cfg)
    
    print('box threshold: %f' % args.box_threshold)
    prompt = [category.strip() for category in re.split(r'\s*\.\s*', args.prompt) if category] + [' ']
    prompt = [[p] for p in prompt]

    model = load_model(args)
    model.eval()
    model.reparameterize(prompt)

    print('loading from', args.ckpt)
    if args.arch == 'enhancer':
        model.enhancer = PromptEnhancer().cuda()
        model.enhancer.eval()
        model.enhancer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'head':
        model.transformer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'bert':
        model.bert.load_state_dict(torch.load(args.ckpt))
    else:
        raise NotImplementedError
    # text_dict = get_text_dict(model, args.prompt)

    if args.dataset == 'scenes100':
        images_train_fewshot, _, images_eval = get_scenes100_images(args)
    if args.dataset == 'egoper':
        _, _, images_eval = get_egoper_images(args)
    if args.dataset == 'hoist':
        _, _, images_eval = get_hoist_images(args)
    
    # images_eval = images_train_fewshot
    loader = torch.utils.data.DataLoader(
        BoxDataset(images_eval, prompt, test_pipeline, training=False),
        batch_size=1, shuffle=False, num_workers=4, collate_fn=BoxDataset.collate
    )

    detections = []
    for batch_inputs, batch_data_samples, im_info in tqdm.tqdm(loader, ascii=True, total=len(images_eval)):
        im = im_info[0]
        im['annotations'] = []
        with torch.no_grad():
            results = model_forward(model, batch_inputs.cuda(), batch_data_samples, args.arch, with_nms=True, keep_logits=False)
        boxes, scores = results[0].bboxes, results[0].scores
        # scores = logits.sigmoid()
        filt_mask = scores > args.box_threshold
        boxes, scores = boxes[filt_mask], scores[filt_mask]
        for box, s in zip(boxes, scores):
            x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': 0, 'score': float(s.item()), 'bbox_mode': BoxMode.XYXY_ABS})
        detections.append(im)

    if args.dataset == 'scenes100':
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs = evaluate_scenes100_masked_fewshot(images_eval, detections)
    if args.dataset == 'egoper' or args.dataset == 'hoist':
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
    
    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.join('./work_dirs')

    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline_cfg[1].type = 'mmyolo.YOLOv5KeepRatioResize'
    test_pipeline_cfg[2].type = 'mmyolo.LetterResize'
    test_pipeline_cfg[4].type = 'mmyolo.LoadText'
    
    test_pipeline = Compose(test_pipeline_cfg)
    
    print('box threshold: %f' % args.box_threshold)
    prompt = [category.strip() for category in re.split(r'\s*\.\s*', args.prompt) if category] + [' ']
    prompt = [[p] for p in prompt]
    
    model = load_model(args)
    model.eval()
    model.reparameterize(prompt)

    print('loading from', args.ckpt)
    if args.arch == 'enhancer':
        model.enhancer = PromptEnhancer().cuda()
        model.enhancer.eval()
        model.enhancer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'head':
        model.transformer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'bert':
        model.bert.load_state_dict(torch.load(args.ckpt))
    else:
        raise NotImplementedError
    # text_dict = get_text_dict(model, args.prompt)

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
        BoxDataset(images_eval, prompt, test_pipeline, training=False),
        batch_size=1, shuffle=False, num_workers=4, collate_fn=BoxDataset.collate
    )

    detections_per_video, images_per_video = {}, {}
    for batch_inputs, batch_data_samples, im_info in tqdm.tqdm(loader, ascii=True, total=len(images_eval)):
        im = im_info[0]
        if not im['video_id'] in images_per_video:
            images_per_video[im['video_id']] = []
            detections_per_video[im['video_id']] = []
        images_per_video[im['video_id']].append(copy.deepcopy(im))
        im['annotations'] = []
        with torch.no_grad():
            results = model_forward(model, batch_inputs.float().cuda(), batch_data_samples, args.arch, with_nms=True, keep_logits=False)
            boxes, scores = results[0].bboxes, results[0].scores
        filt_mask = scores > args.box_threshold
        boxes, scores = boxes[filt_mask], scores[filt_mask]
        for box, s in zip(boxes, scores):
            x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': 0, 'score': float(s.item()), 'bbox_mode': BoxMode.XYXY_ABS})
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
                    if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                        x1, y1, w, h = ann['bbox']
                        x2, y2 = x1 + w, y1 + h
                        ann['bbox_mode'] = BoxMode.XYXY_ABS
                        ann['bbox'] = [x1, y1, x2, y2]
                    if ann['bbox_mode'] != BoxMode.XYXY_ABS:
                        raise 'box modes %s unrecognized: %s' % (list(BoxMode), ann['bbox_mode'])
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
        print(f1, f1.max(), t)

        _WH = max(_images[-1]['width'], _images[-1]['height'])
        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=int(_WH / 40))
        im = skimage.io.imread(_images[-1]['file_name'])
        pred = list(filter(lambda x: x['score'] > t, _detections[-1]['annotations']))
        skimage.io.imsave(os.path.join(outputdir, '%s_%s_unseen.jpg' % (args.dataset, video_id)), draw_bbox(im, pred, 'Unseen Adapted', font, math.ceil(_WH / 500)), quality=90)

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
    if args.arch == 'enhancer':
        model.enhancer = PromptEnhancer().cuda()
        model.enhancer.eval()
        model.enhancer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'head':
        model.transformer.load_state_dict(torch.load(args.ckpt))
    elif args.arch == 'bert':
        model.bert.load_state_dict(torch.load(args.ckpt))
    else:
        raise NotImplementedError
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
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': args.inclusive_class, 'score': float(s), 'bbox_mode': BoxMode.XYXY_ABS})
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
                if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                    x1, y1, w, h = ann['bbox']
                    x2, y2 = x1 + w, y1 + h
                    ann['bbox_mode'] = BoxMode.XYXY_ABS
                    ann['bbox'] = [x1, y1, x2, y2]
                if ann['bbox_mode'] != BoxMode.XYXY_ABS:
                    raise 'box modes %s unrecognized: %s' % (list(BoxMode), ann['bbox_mode'])
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
    if args.arch == 'enhancer':
        model.enhancer = PromptEnhancer().cuda()
        model.enhancer.eval()
        model.enhancer.load_state_dict(torch.load(args.ckpt))
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
    parser = argparse.ArgumentParser('YOLO-World example', add_help=True)
    parser.add_argument('--opt', type=str)
    parser.add_argument('--config', '-c', type=str, default='configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py', help='path to config file')
    parser.add_argument('--box_threshold', type=float, default=0.00, help='box threshold')
    parser.add_argument('--model_ckpt', '-p', type=str, default='yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth', help='path to checkpoint file')
    parser.add_argument('--mask', action='store_true', help='masked data for Scenes100 or not')

    parser.add_argument('--dataset', type=str, default=None, choices=['scenes100', 'egoper', 'hoist'])
    parser.add_argument('--egoper_dir', type=str, default='../../../PTG/ptg_detection')
    parser.add_argument('--hoist_dir', type=str, default='../../../OIH_VIS')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--inclusive_class', type=int, default=0, choices=[0, 1])
    parser.add_argument('--shot', type=int, default=1, help='few-shot learning')
    parser.add_argument('--setting', type=str, default='unseen', choices=['unseen', 'seen'])

    parser.add_argument('--arch', type=str, default='enhancer', choices=['enhancer', 'head', 'bert'])
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
python finetune_yolow_accv.py --opt train --iters 4000 --eval_interval 501 --image_batch_size 32 --num_workers 2 --arch enhancer --dataset scenes100 --shot 1 --prompt "person . vehicle ." --setting seen
python finetune_yolow_accv.py --opt eval --arch enhancer --dataset scenes100 --shot 1 --prompt "person . vehicle ." --ckpt gdinoswint_scenes100_1shot_seen_enhancer.pth

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

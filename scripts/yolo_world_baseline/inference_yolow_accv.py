import argparse
import os
import sys
import json
import tqdm
import copy
import contextlib
import cv2
import re 

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch
from detectron2.structures import BoxMode

# import yolo_world
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmdet.structures.bbox import HorizontalBoxes

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation import eval_AP

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # ERROR: huggingface/tokenizers: The current process just got forked, after parallelism has already been used.


def load_model(args):
    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.join('./work_dirs')
    # init model
    cfg.load_from = args.model_ckpt
    model = init_detector(cfg, checkpoint=args.model_ckpt, device='cuda:0')
    return model


def get_output_no_classify(model, batch_inputs, batch_data_samples):
    data_batch = dict(inputs=batch_inputs,
                      data_samples=batch_data_samples)
    data_batch = model.data_preprocessor(data_batch, training=False)
    img_feats, txt_feats = model.extract_feat(data_batch['inputs'],
                                                data_batch['data_samples'])

    model.bbox_head.num_classes = txt_feats[0].shape[0]
    results_list = model.bbox_head.predict(img_feats,
                                            txt_feats,
                                            data_batch['data_samples'],
                                            rescale=True)

    result = results_list[0]
    return result.bboxes, result.scores


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, images, prompt, transform, training=True):
        super(BoxDataset, self).__init__()
        self.images = images
        self.transform = transform
        self.training = training
        self.prompt = prompt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im_i = self.images[i]

        image = cv2.imread(im_i['file_name'])
        image = image[:, :, [2, 1, 0]]

        if not self.training:
            data_info = dict(img=image, img_id=i, texts=self.prompt)
            data_info = self.transform(data_info)   
            return data_info, copy.deepcopy(im_i)    

        targets = {'labels': [], 'boxes': []}
        for ann in im_i['annotations']:
            if ann['bbox_mode'] == BoxMode.XYXY_ABS:
                x1, y1, x2, y2 = ann['bbox']
                xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            else:
                assert ann['bbox_mode'] == BoxMode.XYWH_ABS
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

        data_info = dict(img=image, img_id=i, texts=self.prompt)

        data_info = self.transform(data_info)
        return data_info, targets

    @staticmethod
    def collate(batch):
        batch_inputs = torch.stack([sample[0]['inputs'] for sample in batch])
        batch_data_samples = [sample[0]['data_samples'] for sample in batch]
        # breakpoint()
        targets = [sample[1] for sample in batch]  
        return batch_inputs, batch_data_samples, targets


def get_scenes100_images(args, keep_class=False):
    import hashlib
    video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
    video_id_list = sorted(video_id_list, key=lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
    video_id_list_train, video_id_list_test = map(sorted, [video_id_list[: 50], video_id_list[50 :]])
    print('training videos:', ' '.join(video_id_list_train))
    print('testing videos: ', ' '.join(video_id_list_test))
    data_type = 'masked' if args.mask else 'unmasked'
    images_train_fewshot, images_test_fewshot, images_eval = [], [], []
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        images = sorted(images, key=lambda x: x['file_name'])
        for im in images:
            im['video_id'] = video_id
            im['file_name'] = os.path.join(inputdir, data_type, im['file_name'])
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
    print('annotation: %d images %d bboxes' % (len(images), sum(map(lambda x: len(x['annotations']), images))))

    for im in detections:
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
    print('detections: %d images %d bboxes' % (len(detections), sum(map(lambda x: len(x['annotations']), detections))))
    return eval_AP(images, detections, return_thres=False)


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
                ann['bbox_mode'] = BoxMode.XYXY_ABS
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
                ann['bbox_mode'] = BoxMode.XYXY_ABS
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

    torch.cuda.empty_cache()
    if args.dataset == 'scenes100':
        _, _, images_eval = get_scenes100_images(args)
    if args.dataset == 'egoper':
        _, _, images_eval = get_egoper_images(args)
    if args.dataset == 'hoist':
        _, _, images_eval = get_hoist_images(args)

   
    loader = torch.utils.data.DataLoader(
        BoxDataset(images_eval, prompt, test_pipeline, training=False),
        batch_size=1, shuffle=False, num_workers=4, collate_fn=BoxDataset.collate
    )

    detections = []
    for batch_inputs, batch_data_samples, im_info in tqdm.tqdm(loader, ascii=True, total=len(images_eval)):
        im = im_info[0]
        im['annotations'] = []
        with torch.no_grad():
            boxes, scores = get_output_no_classify(model, batch_inputs.float().cuda(), batch_data_samples)
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
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': args.inclusive_class, 'score': float(s), 'bbox_mode': BoxMode.XYXY_ABS})
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
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS
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
            im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': 0, 'score': float(s), 'bbox_mode': BoxMode.XYXY_ABS})
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

    torch.cuda.empty_cache()
    if args.dataset == 'scenes100':
        _, _, images_eval = get_scenes100_images(args)
    if args.dataset == 'egoper':
        _, _, images_eval = get_egoper_images(args)
    if args.dataset == 'hoist':
        _, _, images_eval = get_hoist_images(args)

   
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
            boxes, scores = get_output_no_classify(model, batch_inputs.float().cuda(), batch_data_samples)
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

    outputdir = './case_study/'
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
    parser = argparse.ArgumentParser('YOLO-World example', add_help=True)
    parser.add_argument('--opt', type=str, default='detect')
    parser.add_argument('--config', '-c', type=str, default='configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py', help='path to config file')
    parser.add_argument('--mask', action='store_true', help='masked data for Scenes100 or not')
    parser.add_argument('--model_ckpt', '-p', type=str, default='yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth', help='path to checkpoint file')
    
    parser.add_argument('--box_threshold', type=float, default=0.00, help='box threshold')
    # parser.add_argument('--text_threshold', type=float, default=0.25, help='text threshold')
    parser.add_argument('--inclusive_class', type=int, default=0, choices=[0, 1])

    parser.add_argument('--dataset', type=str, default=None, choices=['scenes100', 'egoper', 'hoist'])
    parser.add_argument('--egoper_dir', type=str, default='../../../PTG/ptg_detection')
    parser.add_argument('--hoist_dir', type=str, default='../../../OIH_VIS')
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

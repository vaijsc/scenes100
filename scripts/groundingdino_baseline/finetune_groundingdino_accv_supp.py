import argparse
import os
import sys
import gc
import json
import tqdm
import copy
import contextlib
import hashlib

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

from inference_groundingdino_accv import *
from finetune_groundingdino_accv import *

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # ERROR: huggingface/tokenizers: The current process just got forked, after parallelism has already been used.


def train_few_shot(args):
    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        RandomAdjustSharpnessWrapper(),
        ColorJitterWrapper(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('box threshold: %f' % args.box_threshold)
    model = load_model(args.config).cuda()
    model.eval()
    num_model_params = count_parameters(model)
    for p in model.parameters():
        p.requires_grad = False   

    gc.collect()
    torch.cuda.empty_cache()

    loss_fn = SetCriterion(
        matcher     = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, focal_alpha=0.25),
        focal_alpha = 0.25,
        focal_gamma = 2,
        losses      = ['labels', 'boxes']
    )
    loss_history_dict = {'loss_ce': [], 'loss_bbox': [], 'loss_giou': [], 'weighted': []}

    assert args.setting == 'unseen'
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

    images_per_video = {}
    for im in images_fewshot:
        if not im['video_id'] in images_per_video:
            images_per_video[im['video_id']] = []
        images_per_video[im['video_id']].append(im)
    video_id_list = sorted(list(images_per_video.keys()), key=lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
    print(video_id_list, len(video_id_list))
    video_id_list_divides, N = [], len(video_id_list) // args.divide
    while len(video_id_list) > 0:
        video_id_list_divides.append(video_id_list[:N])
        video_id_list = video_id_list[N:]
    if len(video_id_list_divides[-1]) < N:
        video_id_list_divides[-2] = video_id_list_divides[-2] + video_id_list_divides[-1]
        video_id_list_divides = video_id_list_divides[:-1]
    print('divides:', [len(x) for x in video_id_list_divides])

    for i_divide, video_ids_per_divide in enumerate(video_id_list_divides):
        if args.arch == 'lora':
            del model
            model = load_model(args.config).cuda()
            model.eval()
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
            del model
            model = load_model(args.config).cuda()
            model.eval()
            insert_losa(model, args)

            if args.ckpt is not None:
                print('loading from', args.ckpt)
                losa_state_dict = torch.load(args.ckpt)
                model.transformer.encoder.losa.load_state_dict(losa_state_dict)

            model.transformer.encoder.losa.train()
            optimizer = torch.optim.Adam(model.transformer.encoder.losa.parameters(), lr=args.lr)
            print(f"LoSA params take {count_parameters(model.transformer.encoder.losa)/num_model_params*100:.2f}%")

        elif args.arch == 'res_tuner':
            del model
            model = load_model(args.config).cuda()
            model.eval()
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
            print(f"ResTuner params take {count_parameters(model.tuner_layers)/num_model_params*100:.2f}%")
            
            with torch.no_grad():
                text_dict_cache = get_text_dict(model, args.prompt)
        
        elif args.arch == 'bitfit':
            del model
            model = load_model(args.config).cuda()
            model.eval()
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
            with torch.no_grad():
                text_dict_cache = get_text_dict(model, args.prompt)
                 
            for key in text_dict_cache:
                if key != 'caption':
                    text_dict_cache[key] = text_dict_cache[key].detach()
            # Zero-init
            text_dict_cache['encoded_text'] = torch.zeros_like(text_dict_cache['encoded_text'], requires_grad=True)
            
            # Init from encoded text
            # text_dict_cache['encoded_text'].requires_grad = True

            if args.ckpt is not None:
                print('loading from', args.ckpt)
                text_dict_cache['encoded_text'] = torch.load(args.ckpt)

            optimizer = torch.optim.Adam([text_dict_cache['encoded_text']], lr=args.lr)
            print(f"Prompt params take {text_dict_cache['encoded_text'].numel()/num_model_params*100:.2f}%")

        elif args.arch == 'enhancer':
            model.enhancer = PromptEnhancer(args.enhancer_r_qkv, args.enhancer_r_ff).cuda()
            if args.ckpt is not None:
                print('loading from', args.ckpt)
                model.enhancer.load_state_dict(torch.load(args.ckpt))
            model.enhancer.train()
            optimizer = torch.optim.Adam(model.enhancer.parameters(), lr=args.lr)

            with torch.no_grad():
                text_dict_cache = get_text_dict(model, args.prompt)
            
            print(f"Enhancer params take {count_parameters(model.enhancer)/num_model_params*100:.2f}%")

        elif args.arch == 'head':
            del model
            model = load_model(args.config).cuda()
            model.eval()
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

        elif args.arch == 'linear':
            del model
            model = load_model(args.config).cuda()
            model.eval()
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
        
        elif args.arch == 'bert':
            del model
            model = load_model(args.config).cuda()
            model.eval()
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

        images_per_divide = []
        for v in video_ids_per_divide:
            images_per_divide.extend(images_per_video[v])
        dst_train = BoxDataset(images_per_divide, image_transform)
        loader_train = torch.utils.data.DataLoader(dst_train, batch_size=args.image_batch_size, collate_fn=BoxDataset.collate, shuffle=True, num_workers=args.num_workers)
        iter_train = iter(loader_train)
        loss_history_dict = {'loss_ce': [], 'loss_bbox': [], 'loss_giou': [], 'weighted': []}
        for it in tqdm.tqdm(range(0, args.iters // args.divide), ascii=True, desc='%d: %d images' % (i_divide, len(images_per_divide))):
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
                if targets['labels'].size(0) < 1:
                    continue
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
            if len(targets_all) < 1:
                continue
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

            if it % args.eval_interval == 0 or it == args.iters // args.divide - 1:
                if len(loss_history_dict['weighted']) > 40:
                    plt.figure(figsize=(20, 20))
                    loss_key_list = ['loss_ce', 'loss_bbox', 'loss_giou', 'weighted']
                    loss_tensor = torch.tensor([loss_history_dict[k] for k in loss_key_list]).float()
                    kernel = max(min(100, len(loss_history_dict['weighted']) // 20), 20)
                    loss_smooth = torch.nn.functional.avg_pool1d(loss_tensor.unsqueeze(1), kernel)
                    loss_smooth = loss_smooth[:, 0, :].detach().numpy()
                    for i, loss_key in enumerate(loss_key_list):
                        plt.subplot(2, 2, i + 1)
                        plt.plot(np.arange(0, loss_smooth.shape[1]) * kernel, loss_smooth[i], 'r-')
                        plt.grid(True)
                        plt.xlim(0, args.iters // args.divide)
                        plt.xlabel('Training Iterations')
                        plt.title(loss_key)
                    plt.tight_layout()
                    plt.savefig(prefix + '.%03d.pdf' % i_divide)
                    plt.close()
        if args.arch == 'lora':
            lora_state_dict = {}
            for i, lora_layer in enumerate(model.lora_layers):
                # Save each LoRA layer's state dict
                lora_state_dict[f"lora_{i}"] = lora_layer.state_dict()
            torch.save(lora_state_dict, prefix + '.%03d.pth' % i_divide)
        elif args.arch == 'losa':
            torch.save(model.transformer.encoder.losa.state_dict(), prefix + '.%03d.pth' % i_divide)
        elif args.arch == 'res_tuner':
            res_tuner_state_dict = {}
            for i, tuner_layer in enumerate(model.tuner_layers):
                # Save each ResTuner layer's state dict
                res_tuner_state_dict[f"tuner_{i}"] = tuner_layer.state_dict()
            torch.save(res_tuner_state_dict, prefix + '.%03d.pth' % i_divide)
        elif args.arch == 'bitfit':
            bias_dict = {name: param for name, param in model.named_parameters() if 'bias' in name and 'bert' in name}
            torch.save(bias_dict, prefix + '.%03d.pth' % i_divide)
        elif args.arch == 'prompt':
            torch.save(text_dict_cache['encoded_text'], prefix + '.%03d.pth' % i_divide)
        elif args.arch == 'enhancer':
            torch.save(model.enhancer.state_dict(), prefix + '.%03d.pth' % i_divide)
        elif args.arch == 'head':
            torch.save(model.transformer.state_dict(), prefix + '.%03d.pth' % i_divide)
        elif args.arch == 'linear':
            linear_state_dict = {'bbox': model.transformer.decoder.bbox_embed.state_dict(), 'class': model.transformer.decoder.class_embed.state_dict()}
            torch.save(linear_state_dict, prefix + '.%03d.pth' % i_divide)
        elif args.arch == 'bert':
            torch.save(model.bert.state_dict(), prefix + '.%03d.pth' % i_divide)
        else:
            raise NotImplementedError


def show_divide_AP(args):
    def _plot(APs, ymin, ymax, yticks, f):
        # for d in APs:
        #     assert d == len(APs[d])
        plt.figure(figsize=(5, 5))
        APs = [(d, np.array(APs[d]).mean(), np.array(APs[d]).std()) for d in sorted(list(APs.keys()))]
        plt.plot(np.arange(0, len(APs)), [x for _, x, _ in APs], 'ro-', linewidth=2, markersize=8)
        plt.errorbar(np.arange(2, len(APs)), [x for _, x, _ in APs[2:]], yerr=[x for _, _, x in APs[2:]], fmt='', linewidth=0, markersize=0, elinewidth=2, ecolor='b', capsize=10, capthick=2)
        plt.xlim(-0.25, len(APs) - 0.75)
        plt.xticks(ticks=np.arange(0, len(APs)), labels=['vanilla\nmodel', 'no\ndivide'] + ['%d-\ndivides' % d for d, _, _ in APs[2:]])
        plt.ylim(ymin, ymax)
        plt.yticks(ticks=yticks)
        plt.ylabel('$AP^m$')
        plt.tight_layout()
        plt.grid(True)
        # plt.show()
        plt.savefig(f)

    # scenes100
    APs = {
        0:  [36.60],
        1:  [55.16],
        2:  [55.02, 54.71],
        4:  [54.14, 54.40, 49.98, 54.47],
        8:  [51.66, 49.75, 54.37, 52.95, 51.07, 49.43, 53.20, 54.21]
    }
    _plot(APs, 36, 56, [40, 45, 50, 55], 'gdino_scenes100_divides.pdf')

    # egoper
    APs = {
        0:  [45.99],
        1:  [69.56],
        2:  [67.28, 66.80],
        4:  [64.24, 64.87, 66.24, 65.52],
        8:  [64.46, 62.51, 58.38, 62.38, 63.56, 62.22, 65.83, 66.01],
    }
    _plot(APs, 45, 71, [50, 55, 60, 65, 70], 'gdino_egoper_divides.pdf')

    # hoist
    APs = {
        0:  [22.73],
        1:  [36.51],
        # 10: (np.array([0.32726,0.32731,0.31106,0.34075,0.34395,0.36211,0.32684,0.35075,0.33757,0.33594])*100).tolist(),
        25: (np.array([0.29557,0.27489,0.26135,0.30054,0.28060,0.28213,0.29496,0.26530,0.28040,0.28813,0.32666,0.29380,0.27437,0.26443,0.29667,0.29227,0.29626,0.28609,0.28330,0.30986,0.29491,0.31255,0.29970,0.29927,0.27435])*100).tolist(),
        50: (np.array([0.29029,0.28214,0.23220,0.27232,0.22725,0.25359,0.25569,0.13625,0.28498,0.28164,0.24844,0.29361,0.29850,0.27082,0.24263,0.28339,0.29652,0.28650,0.27383,0.28009,0.26558,0.31068,0.27808,0.30934,0.23880,0.24916,0.23861,0.31506,0.25799,0.26677,0.28441,0.23269,0.20362,0.26010,0.25132,0.27671,0.31629,0.28647,0.28504,0.25545,0.28062,0.22226,0.27150,0.29745,0.26634,0.27235,0.27758,0.28818,0.26025,0.30893])*100).tolist(),
        100:(np.array([0.23289,0.27599,0.28491,0.24931,0.17801,0.22202,0.27090,0.25175,0.24071,0.23361,0.23530,0.27892,0.20379,0.27096,0.25831,0.20184,0.26831,0.27719,0.22798,0.16908,0.24381,0.25842,0.26686,0.15829,0.24767,0.28433,0.23907,0.20349,0.27424,0.21536,0.24593,0.29258,0.21918,0.23199,0.29597,0.25820,0.24915,0.26327,0.20196,0.24638,0.24720,0.22507,0.24851,0.28864,0.25710,0.21854,0.21443,0.24088,0.23061,0.25669,0.27540,0.25110,0.25833,0.22330,0.02016,0.23156,0.20957,0.27280,0.25235,0.23086,0.26372,0.26607,0.26338,0.24732,0.16802,0.30292,0.21567,0.25192,0.19127,0.24789,0.24826,0.22832,0.30558,0.21597,0.29332,0.26672,0.25717,0.28648,0.22667,0.24681,0.26315,0.24093,0.22599,0.25657,0.24923,0.14093,0.26959,0.19369,0.27178,0.26715,0.20213,0.24299,0.20878,0.19010,0.26188,0.27707,0.23950,0.22143,0.26764,0.25223])*100).tolist()
    }
    _plot(APs, 20, 37, [25, 30, 35], 'gdino_hoist_divides.pdf')


def train_faster_rcnn(args):
    import types
    import skimage.io
    import detectron2
    from detectron2 import model_zoo
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.engine.train_loop import SimpleTrainer
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models import get_cfg_base_model
    from base_detector_train import get_cfg, FinetuneTrainer, simple_trainer_run_step

    assert args.setting == 'unseen'
    if args.dataset == 'ovcoco':
        images_fewshot, images_eval = get_ovcoco_images(args)
    elif args.dataset == 'rareplanes':
        images_fewshot, images_eval = get_rareplanes_images(args)
    else:
        if args.dataset == 'scenes100':
            images_train_fewshot, images_test_fewshot, images_eval = get_scenes100_images(args)
        if args.dataset == 'egoper':
            images_train_fewshot, images_test_fewshot, images_eval = get_egoper_images(args)
        if args.dataset == 'hoist':
            images_train_fewshot, images_test_fewshot, images_eval = get_hoist_images(args)
        if args.dataset == 'birdsai':
            images_train_fewshot, images_test_fewshot, images_eval = get_birdsai_images(args)
        images_fewshot = images_train_fewshot
        
    print('%s %d-shot %s training set: %d images' % (args.dataset, args.shot, args.setting, len(images_fewshot)))
    prefix = os.path.join(os.path.dirname(__file__), 'fasterrcnn_%s_%dshot_%s' % (args.dataset, args.shot, args.setting))
    print(prefix)

    desc = 'train_%s_%dshot_%s' % (args.dataset, args.shot, args.setting)
    DatasetCatalog.register(desc, lambda: images_fewshot)
    MetadataCatalog.get(desc).thing_classes = ['object']

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    cfg.OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'finetune_output_%s' % args.dataset)
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.GAMMA = 1
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD = args.iters + 1
    cfg.DATASETS.TRAIN = (desc,)
    cfg.DATASETS.TEST = []
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    print('- image batch size:', cfg.SOLVER.IMS_PER_BATCH)
    print('- roi batch size:', cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)
    print('- base lr:', cfg.SOLVER.BASE_LR)

    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200
    trainer = FinetuneTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(simple_trainer_run_step, trainer._trainer)
    trainer._trainer.split_batch = -1
    trainer.resume_or_load(resume=False)
    trainer.train()
    torch.save(trainer.model.state_dict(), prefix + '.pth')

    class EvaluationDataset(torch.utils.data.Dataset):
        def __init__(self, images):
            super(EvaluationDataset, self).__init__()
            self.images = images
        def __len__(self):
            return len(self.images)
        def __getitem__(self, i):
            im = skimage.io.imread(self.images[i]['file_name'])
            if len(im.shape) == 2:
                im = np.stack([im] * 3, axis=2)
            return self.images[i], im[:, :, ::-1]
        @staticmethod
        def collate(batch):
            return batch

    del trainer
    cfg.MODEL.WEIGHTS = prefix + '.pth'
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    detector = DefaultPredictor(cfg)
    detections = []
    loader = torch.utils.data.DataLoader(EvaluationDataset(images_eval), batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=2)
    for im, im_arr in tqdm.tqdm(loader, total=len(images_eval), ascii=True):
        det = copy.deepcopy(im)
        det['annotations'] = []
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': 0, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)

    if args.dataset == 'scenes100':
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs = evaluate_scenes100_masked_fewshot(images_eval, detections)
    else:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs = eval_AP(images_eval, detections, return_thres=False)
    print(APs['results'])


def train_yolov8(args):
    import math
    import shutil
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils.checks import check_imgsz
    from ultralytics.models.yolo.detect.predict import DetectionPredictor

    assert args.setting == 'unseen'
    if args.dataset == 'ovcoco':
        images_fewshot, images_eval = get_ovcoco_images(args)
    elif args.dataset == 'rareplanes':
        images_fewshot, images_eval = get_rareplanes_images(args)
    else:
        if args.dataset == 'scenes100':
            images_train_fewshot, images_test_fewshot, images_eval = get_scenes100_images(args)
        if args.dataset == 'egoper':
            images_train_fewshot, images_test_fewshot, images_eval = get_egoper_images(args)
        if args.dataset == 'hoist':
            images_train_fewshot, images_test_fewshot, images_eval = get_hoist_images(args)
        if args.dataset == 'birdsai':
            images_train_fewshot, images_test_fewshot, images_eval = get_birdsai_images(args)
        images_fewshot = images_train_fewshot
    print('%s %d-shot %s training set: %d images' % (args.dataset, args.shot, args.setting, len(images_fewshot)))
    prefix = os.path.join(os.path.dirname(__file__), 'yolov8l_%s_%dshot_%s' % (args.dataset, args.shot, args.setting))
    print(prefix)
    epochs = math.ceil(args.image_batch_size * args.iters / len(images_fewshot))
    print('train for %d epochs' % epochs)

    yolo_dst_path = '/tmp/yolov8_dataset'
    assert not os.access(yolo_dst_path, os.R_OK)
    os.mkdir(yolo_dst_path)
    os.mkdir(os.path.join(yolo_dst_path, 'images'))
    os.mkdir(os.path.join(yolo_dst_path, 'images', 'train'))
    os.mkdir(os.path.join(yolo_dst_path, 'labels'))
    os.mkdir(os.path.join(yolo_dst_path, 'labels', 'train'))
    for im in tqdm.tqdm(images_fewshot, ascii=True):
        fn = im['video_id'] + '_' + os.path.basename(im['file_name'])
        shutil.copyfile(im['file_name'], os.path.join(yolo_dst_path, 'images', 'train', fn))
        imW, imH = im['width'], im['height']
        fp = open(os.path.join(yolo_dst_path, 'labels', 'train', fn[:-3] + 'txt'), 'w')
        for ann in im['annotations']:
            assert ann['bbox_mode'] == 0
            x1, y1, x2, y2 = ann['bbox']
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            xc, yc, w, h = xc / imW, yc / imH, w / imW, h / imH
            # class x_center y_center width height
            fp.write('0 %.5f %.5f %.5f %.5f\n' % (xc, yc, w, h))
        fp.close()
    yolo_dst_yaml = os.path.join(yolo_dst_path, 'dataset.yaml')
    with open(yolo_dst_yaml, 'w') as fp:
        fp.write('train: %s\nval: %s\nnc: 1\nnames: [\'object\']' % (os.path.join(yolo_dst_path, 'images', 'train'), os.path.join(yolo_dst_path, 'images', 'train')))

    imgsz = 1312
    model = YOLO('yolov8l.pt', verbose=True)
    model.train(
        data          = yolo_dst_yaml,
        imgsz         = imgsz,
        epochs        = epochs,
        warmup_epochs = 0,
        batch         = args.image_batch_size,
        optimizer     = 'SGD',
        momentum      = 0.9,
        lr0           = args.lr,
        lrf           = 1,
        workers       = args.image_batch_size,
        resume        = False,
        amp           = False,
        show_labels   = False,
        plots         = False,
        save_json     = False,
    )

    predictor_args = {
        'model': model,
        'task': 'detect',
        'mode': 'predict',
        'imgsz': imgsz,
        'conf': 0.05,
        'batch': 1,
        'save': False,
    }
    predictor = DetectionPredictor(overrides=predictor_args)
    predictor.setup_model(model=model.model)
    predictor.imgsz = check_imgsz(imgsz, stride=model.model.stride, min_dim=2)

    detections = []
    predictor.batch = [[0]]
    with torch.no_grad():
        for im in tqdm.tqdm(images_eval, ascii=True):
            det = copy.deepcopy(im)
            det['annotations'] = []
            im_arr = cv2.imread(im['file_name'], cv2.IMREAD_COLOR) # cv2 reads as BGR, YOLOv8 also uses BGR
            im_pp = predictor.preprocess([im_arr])
            preds = model.model(im_pp, augment=False, visualize=False, embed=None)
            boxes = predictor.postprocess(preds, im_pp, [im_arr])[0].boxes
            for conf, xyxy in zip(boxes.cls, boxes.xyxy):
                det['annotations'].append({'bbox': list(map(float, xyxy)), 'bbox_mode': 0, 'segmentation': [], 'category_id': 0, 'score': float(conf)})
            detections.append(det)

    if args.dataset == 'scenes100':
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs = evaluate_scenes100_masked_fewshot(images_eval, detections)
    else:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs = eval_AP(images_eval, detections, return_thres=False)
    print(APs['results'])


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
    parser.add_argument('--divide', type=int, default=0, help='num divides')

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
    if args.opt == 'show_divide':
        show_divide_AP(args)
    if args.opt == 'frcnn':
        train_faster_rcnn(args)
    if args.opt == 'yolo':
        train_yolov8(args)

'''
python finetune_groundingdino_accv_supp.py --opt fewshot --iters 2000 --image_batch_size 2 --num_workers 2 --arch enhancer --dataset scenes100 --shot 1 --prompt "person . vehicle ." --setting unseen --divide 2
for F in `find /mnt/f/intersections_results/accv24/gdino/Scenes100_person_vehicle/divide* | grep pth` ; do python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset scenes100 --shot 1 --prompt "person . vehicle ." --ckpt ${F} ; done

python finetune_groundingdino_accv_supp.py --opt fewshot --iters 2000 --image_batch_size 2 --num_workers 2 --arch enhancer --dataset egoper --shot 1 --prompt "kitchen object ." --setting unseen --divide 2
for F in `find /mnt/f/intersections_results/accv24/gdino/EgoPER_kitchenobject/divide* | grep pth` ; do python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset egoper --shot 1 --prompt "kitchen object ." --ckpt ${F} ; done

python finetune_groundingdino_accv_supp.py --opt fewshot --iters 20000 --image_batch_size 2 --num_workers 2 --arch enhancer --dataset hoist --shot 1 --prompt "hand-held object ." --setting unseen --divide 2


for F in `ls gdinoswint_hoist_1shot_25divide_unseen_enhancer.*pth` ; do CUDA_VISIBLE_DEVICES=4 python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset hoist --shot 1 --prompt "hand-held object ." --ckpt ${F} --hold 16 ; done
nohup bash gdino_eval_d25.sh &> log.eval.hoist.d25.txt &
cat log.eval.hoist.d25.txt | grep overall | wc -l
cat log.eval.hoist.d25.txt | grep overall | cut -d'[' -f 2 | cut -d',' -f1 | cut -c-7 | paste -sd "," -

for F in `ls gdinoswint_hoist_1shot_50divide_unseen_enhancer.*pth | head -n25` ; do CUDA_VISIBLE_DEVICES=5 python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset hoist --shot 1 --prompt "hand-held object ." --ckpt ${F} --hold 16 ; done
nohup bash gdino_eval_d50_1.sh &> log.eval.hoist.d50.1.txt &
for F in `ls gdinoswint_hoist_1shot_50divide_unseen_enhancer.*pth | tail -n25` ; do CUDA_VISIBLE_DEVICES=6 python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset hoist --shot 1 --prompt "hand-held object ." --ckpt ${F} --hold 16 ; done
nohup bash gdino_eval_d50_2.sh &> log.eval.hoist.d50.2.txt &
cat log.eval.hoist.d50*txt | grep overall | wc -l
cat log.eval.hoist.d50*txt | grep overall | cut -d'[' -f 2 | cut -d',' -f1 | cut -c-7 | paste -sd "," -

for F in `ls gdinoswint_hoist_1shot_100divide_unseen_enhancer.*pth | head -n25` ; do CUDA_VISIBLE_DEVICES=7 python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset hoist --shot 1 --prompt "hand-held object ." --ckpt ${F} --hold 16 ; done
nohup bash gdino_eval_d100_1.sh &> log.eval.hoist.d100.1.txt &
for F in `ls gdinoswint_hoist_1shot_100divide_unseen_enhancer.*pth | head -n50 | tail -n25` ; do CUDA_VISIBLE_DEVICES=3 python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset hoist --shot 1 --prompt "hand-held object ." --ckpt ${F} --hold 16 ; done
nohup bash gdino_eval_d100_2.sh &> log.eval.hoist.d100.2.txt &
for F in `ls gdinoswint_hoist_1shot_100divide_unseen_enhancer.*pth | head -n75 | tail -n25` ; do CUDA_VISIBLE_DEVICES=2 python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset hoist --shot 1 --prompt "hand-held object ." --ckpt ${F} --hold 16 ; done
nohup bash gdino_eval_d100_3.sh &> log.eval.hoist.d100.3.txt &
for F in `ls gdinoswint_hoist_1shot_100divide_unseen_enhancer.*pth | tail -n28` ; do CUDA_VISIBLE_DEVICES=1 python finetune_groundingdino_accv.py --opt eval --arch enhancer --dataset hoist --shot 1 --prompt "hand-held object ." --ckpt ${F} --hold 16 ; done
nohup bash gdino_eval_d100_4.sh &> log.eval.hoist.d100.4.txt &
cat log.eval.hoist.d100*txt | grep overall | wc -l
cat log.eval.hoist.d100*txt | grep overall | cut -d'[' -f 2 | cut -d',' -f1 | cut -c-7 | paste -sd "," -

python finetune_groundingdino_accv_supp.py --opt show_divide


python finetune_groundingdino_accv_supp.py --opt frcnn --iters 2000 --image_batch_size 2 --num_workers 2 --dataset scenes100 --shot 1 --setting unseen
python finetune_groundingdino_accv_supp.py --opt frcnn --iters 2000 --image_batch_size 2 --num_workers 2 --dataset egoper --shot 1 --setting unseen
python finetune_groundingdino_accv_supp.py --opt frcnn --iters 20000 --image_batch_size 2 --num_workers 2 --dataset hoist --shot 1 --setting unseen


python finetune_groundingdino_accv_supp.py --opt yolo --iters 2000 --image_batch_size 2 --num_workers 2 --dataset scenes100 --shot 1 --setting unseen
python finetune_groundingdino_accv_supp.py --opt yolo --iters 2000 --image_batch_size 2 --num_workers 2 --dataset egoper --shot 1 --setting unseen
python finetune_groundingdino_accv_supp.py --opt yolo --iters 20000 --image_batch_size 2 --num_workers 2 --dataset hoist --shot 1 --setting unseen
'''
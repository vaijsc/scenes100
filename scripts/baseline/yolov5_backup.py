# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from typing import Dict, List, Optional, Tuple
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from common import *  # noqa
from experimental import *  # noqa
from yolov5_utils.autoanchor import check_anchor_order
from yolov5_utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args, non_max_suppression, scale_boxes, xywhn2xyxy, xyxy2xywhn
from yolov5_utils.plots import feature_visualization
from yolov5_utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
from yolov5_utils.loss import *
from yolov5_utils.augmentations import letterbox
from yolov3_utils.augmentations import *

import detectron2
from detectron2.structures import Instances, Boxes

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=2, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward_image(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')
        self = self.cuda()
        self.compute_loss = ComputeLoss(self)
        if not self.training:
            self.input_size = (800, 1344) # have to be divisible by 32
        else:
            self.input_size = (640, 1088)
        
        self.output_format = 'frcnn'


    def reverse_yolo_transform(self, yolo_outputs, batched_inputs):
        reversed_outputs = []
        for i, output in enumerate(yolo_outputs):
            pad = batched_inputs[i]['pad']
            ratio = batched_inputs[i]['ratio']
            h_old, w_old = batched_inputs[i]['old_shape'] # shape before padding

            scores = output[:, 4].cuda()
            pred_classes = output[:, 5].cuda()
            reversed_boxes = torch.zeros((output.shape[0], 4))
            reversed_boxes[:, [0, 2]] = (reversed_boxes[:, [0, 2]] - pad[0])/ratio[0]
            reversed_boxes[:, [1, 3]] = (reversed_boxes[:, [1, 3]] - pad[1])/ratio[1]
            # mask = (w_old >= reversed_boxes[:, 0] and \
            #         w_old >= reversed_boxes[:, 2] and \
            #         h_old >= reversed_boxes[:, 1] and \
            #         h_old >= reversed_boxes[:, 3]
            #         )

            boxes = Boxes(reversed_boxes.cuda())
            instances_dict = {'pred_boxes': boxes, 'scores': scores, 'pred_classes': pred_classes}
            instances = Instances(image_size=(h_old, w_old))
            for (k, v) in instances_dict.items():
                instances.set(k, v)
            reversed_outputs.append(instances)
        return reversed_outputs


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        for i, im in enumerate(batched_inputs):
            # BGR to RGB
            im['image'] = torch.from_numpy(im['image'][:3].numpy()[::-1, :, :].copy())
            old_shape = im["image"].shape[1:]
            im['old_shape'] = old_shape
            im['image'] = im['image'].permute(1, 2, 0)
            if self.training:
                self.input_size = (608, 1056)
                h_img, w_img = old_shape[0], old_shape[1]
                boxes_XYXY = im['instances'].gt_boxes.tensor
                boxes_XYWHN = torch.zeros_like(boxes_XYXY)
                boxes_XYWHN[:, 0] = ((boxes_XYXY[:, 0] + boxes_XYXY[:, 2]) / 2) / w_img
                boxes_XYWHN[:, 1] = ((boxes_XYXY[:, 1] + boxes_XYXY[:, 3]) / 2) / h_img
                boxes_XYWHN[:, 2] = (boxes_XYXY[:, 2] - boxes_XYXY[:, 0]) / w_img
                boxes_XYWHN[:, 3] = (boxes_XYXY[:, 3] - boxes_XYXY[:, 1]) / h_img
                im['instances'].gt_boxes.tensor = boxes_XYWHN
                
                labels = im['instances'].gt_classes.reshape(-1, 1)
                instances = torch.cat([labels, im['instances'].gt_boxes.tensor], dim=1)
                # letterbox resize image and label
                im['image'], ratio, pad = letterbox(im['image'].numpy(), new_shape=self.input_size)
                instances[:, 1:] = xywhn2xyxy(x=instances[:, 1:], w=int(w_img*ratio[0]), h=int(h_img*ratio[1]), padw=pad[0], padh=pad[1])
                instances[:, 1:] = xyxy2xywhn(x=instances[:, 1:], w=im['image'].shape[2], h=im['image'].shape[1])

                im['image'] = np.ndarray.astype(im['image'], np.uint8)
                im['image'] = torchvision.transforms.functional.to_tensor(im['image'])
                sample_index = torch.full((instances.shape[0], 1), i, device=instances.device)
                instances = torch.cat([sample_index, instances], dim=1)
                im['instances'] = instances
            else:
                self.input_size = (800, 1344)
                im['image'], ratio, pad = letterbox(im['image'].numpy(), new_shape=self.input_size)
                im['image'] = np.ndarray.astype(im['image'], np.uint8)
                im['image'] = torchvision.transforms.functional.to_tensor(im['image'])
                im['pad'] = pad
                im['ratio'] = ratio
        
        return batched_inputs

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], augment=False, profile=False, visualize=False):
        if not self.training: return self.inference(batched_inputs)
        batched_inputs = self.preprocess_image(batched_inputs)
        x = torch.stack([im['image'] for im in batched_inputs]).cuda()
        targets = torch.cat([im['instances'].cuda() for im in batched_inputs], dim=0).cuda()

        if augment:
            yolo_outputs = self._forward_augment(x)  # augmented inference, None
        yolo_outputs = self._forward_once(x, profile, visualize)  # single-scale inference, train
        loss, loss_components = self.compute_loss(yolo_outputs, targets)

        return loss, loss_components

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], augment=False, profile=False, visualize=False):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        batched_inputs = self.preprocess_image(batched_inputs)
        x = torch.stack([im['image'] for im in batched_inputs]).cuda()

        if augment:
            yolo_outputs = self._forward_augment(x)  # augmented inference, None
        yolo_outputs = self._forward_once(x, profile, visualize)  # single-scale inference, train

        outputs = non_max_suppression(yolo_outputs[0], conf_thres=0.25, iou_thres=0.45)
        # breakpoint()
        reversed_outputs = self.reverse_yolo_transform(outputs, batched_inputs)
        results = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(reversed_outputs, batched_inputs, [im['old_shape'] for im in batched_inputs])
        return results
    
    def forward_image(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    

Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def load_yolov5(model_path, weights_path=None):
    """Loads the yolov5 model from file.

    :param model_path: Path to model definition file (.yaml)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.pth)
    :type weights_path: str
    :return: Returns model
    :rtype: DetectionModel
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    model = Model(model_path).to(device)

    # model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../config/yolov5s.yaml ', help='model.yaml')
    parser.add_argument('--ckpt', type=str, default='../weights/yolov5s_remap.pth', help='weights for model')
    # parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--profile', action='store_true', help='profile model speed')
    # parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    # parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    # opt.cfg = check_yaml(opt.cfg)  # check YAML
    # print_args(vars(opt))
    # device = select_device(opt.device)

    # # Create model
    # im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    # model = Model(opt.cfg).to(device)
    # model.eval()
    # # Options
    # if opt.line_profile:  # profile layer by layer
    #     out = model(im, profile=True)
    #     breakpoint()

    # elif opt.profile:  # profile forward-backward
    #     results = profile(input=im, ops=[model], n=3)

    # elif opt.test:  # test all models
    #     for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
    #         try:
    #             _ = Model(cfg)
    #         except Exception as e:
    #             print(f'Error in {cfg}: {e}')

    # else:  # report fused model summary
    #     model.fuse()


    im = torch.rand(1, 3, 640, 640).to("cuda")
    model = load_yolov5(opt.cfg, opt.ckpt)
    model.train()
    out = model.forward_image(im)
    breakpoint()
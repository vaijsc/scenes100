#!python3

import os
import json
import numpy as np
import skimage.io
import imageio
import time
import tqdm
import lmdb
import argparse
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision
import torch.optim as optim
import torch.nn as nn
import detectron2.layers

#######################################
# all assume images are in RGB format #
#######################################


class IoUEstimator(nn.Module):
    def __init__(self, pretrained=False):
        super(IoUEstimator, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Identity()
        self.regressor = nn.Linear(512, 1)

    def forward(self, X):
        return self.regressor(self.resnet(X))


def ciou_quality(bbox1: torch.Tensor, bbox2: torch.Tensor):
    # https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/layers/losses.py
    return 1.0 - detectron2.layers.losses.ciou_loss(bbox1, bbox2) / 3.0


class MSCOCOBBoxes(data.Dataset):
    def __init__(self, split, cocodir, bbox_per_image):
        super(MSCOCOBBoxes, self).__init__()
        assert split in ['train', 'valid']
        self.bbox_per_image = bbox_per_image
        if split == 'valid':
            annotations_json = os.path.join(cocodir, 'annotations', 'instances_val2017.json')
            images_dir = os.path.join(cocodir, 'images', 'val2017')
        else:
            annotations_json = os.path.join(cocodir, 'annotations', 'instances_train2017.json')
            images_dir = os.path.join(cocodir, 'images', 'train2017')
        with open(annotations_json, 'r') as fp:
            annotations = json.load(fp)

        thing_classes_coco, thing_classes = [['person'], ['car', 'bus', 'truck']], ['person', 'vehicle']
        category_id_remap = {}
        for cat in annotations['categories']:
            for i in range(0, len(thing_classes_coco)):
                if cat['name'] in thing_classes_coco[i]:
                    category_id_remap[cat['id']] = i

        coco_dicts = {}
        for im in annotations['images']:
            coco_dicts[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []}
        for ann in annotations['annotations']:
            if ann['category_id'] in category_id_remap:
                coco_dicts[ann['image_id']]['annotations'].append({'xywh': ann['bbox'], 'area': ann['area'], 'category_id': category_id_remap[ann['category_id']]})
        coco_dicts = list(coco_dicts.values())
        self.coco_dicts = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts))
        count_images, count_bboxes = len(self.coco_dicts), sum(map(lambda ann: len(ann['annotations']), self.coco_dicts))
        print('MSCOCO-2017 %s: %d images, %d bboxes' % (split, count_images, count_bboxes))

        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((160, 160), antialias=True),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.coco_dicts)

    def __getitem__(self, i):
        im_i = self.coco_dicts[i]
        im_arr = skimage.io.imread(im_i['file_name']).astype(np.float32) / 255.0
        if len(im_arr.shape) < 3:
            im_arr = np.stack([im_arr, im_arr, im_arr], axis=2)
        assert len(im_arr.shape) == 3 and im_arr.shape[2] == 3 # RGB
        W0, W1, _ = im_arr.shape

        # sample augmented bboxes
        gt_bboxes, sampled_bboxes = [], []
        for ann in im_i['annotations']:
            x, y, w, h = ann['xywh']
            gt_bboxes.append([x, y, x + w, y + h])
            sampled_bboxes.append(self.augment_bbox(x, y, w, h, W0, W1, 0.2)) # high quality bboxes
            sampled_bboxes.append(self.augment_bbox(x, y, w, h, W0, W1, 1)) # mid quality bboxes
            sampled_bboxes.append(self.augment_bbox(x, y, w, h, W0, W1, -1)) # low quality bboxes
        sampled_bboxes = list(filter(lambda _b: (_b[2] - _b[0]) > 1 and (_b[3] - _b[1]) > 1, sampled_bboxes))
        assert len(sampled_bboxes) > 0
        if len(sampled_bboxes) < self.bbox_per_image:
            sampled_bboxes = sampled_bboxes * self.bbox_per_image
        sampled_bboxes = np.array(sampled_bboxes)
        sampled_bboxes = sampled_bboxes[np.random.choice(np.arange(0, sampled_bboxes.shape[0]), size=self.bbox_per_image)]

        # get highest CIoU quality scores
        sampled_ciou_qualities = []
        gt_bboxes_tensor = torch.tensor(gt_bboxes).float()
        sampled_bboxes_tensor = torch.from_numpy(sampled_bboxes).float()
        for j in range(0, sampled_bboxes_tensor.size(0)):
            sampled_ciou_qualities.append(ciou_quality(gt_bboxes_tensor, sampled_bboxes_tensor[j : j + 1].expand(gt_bboxes_tensor.size(0), -1)).max())

        sampled_bboxes_crop = []
        for x1, y1, x2, y2 in sampled_bboxes:
            sampled_bboxes_crop.append(self.tf(torch.from_numpy(im_arr[y1 : y2, x1 : x2, :].transpose(2, 0, 1)).float()))
        return torch.stack(sampled_bboxes_crop, dim=0), torch.stack(sampled_ciou_qualities, dim=0).float()

    @staticmethod
    def rectify_bbox(x1, y1, x2, y2, W0, W1):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1, x2, y2 = map(lambda _x: max(_x, 0), [x1, y1, x2, y2])
        x2, y2 = min(x2, W1), min(y2, W0)
        return x1, y1, x2, y2

    @staticmethod
    def augment_bbox(x, y, w, h, W0, W1, r):
        if r >= 0:
            x1 = x +     np.random.rand() * w * r - r / 2.0
            y1 = y +     np.random.rand() * h * r - r / 2.0
            x2 = x + w + np.random.rand() * w * r - r / 2.0
            y2 = y + h + np.random.rand() * h * r - r / 2.0
            return MSCOCOBBoxes.rectify_bbox(x1, y1, x2, y2, W0, W1)
        else:
            x1, x2 = sorted((np.random.rand(2,) * W1).tolist())
            y1, y2 = sorted((np.random.rand(2,) * W0).tolist())
            return MSCOCOBBoxes.rectify_bbox(x1, y1, x2, y2, W0, W1)


def train_eval(args):
    np.random.seed(0)
    torch.manual_seed(0)

    assert(torch.cuda.is_available())
    net = IoUEstimator(pretrained=True).cuda()
    for p in net.resnet.parameters():
        p.requires_grad = False
    # loss_fn = nn.BCELoss(reduction='none')
    loss_fn = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 3, gamma=0.1)

    loader_train = data.DataLoader(MSCOCOBBoxes('train', args.cocodir, args.bbox_batch_size // args.image_batch_size), batch_size=args.image_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    loader_valid = data.DataLoader(MSCOCOBBoxes('valid', args.cocodir, args.bbox_batch_size // args.image_batch_size), batch_size=args.image_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    loss_history = []
    for ep in range(0, args.epochs):
        if ep == 2:
            print('unfreeze whole network')
            for p in net.parameters():
                p.requires_grad = True

        loss_history.append({'train': [], 'valid': []})
        net.train()
        for i, (X, y) in enumerate(tqdm.tqdm(loader_train, ascii=True, desc='training   %d/%d' % (ep + 1, args.epochs))):
            # patches, y_np = X.view(64, 3, 160, 160).numpy().transpose(0, 2, 3, 1), y.flatten().numpy()
            # patches -= patches.min(); patches /= patches.max()
            # plt.figure()
            # for j in range(1, 41):
            #     plt.subplot(5, 8, j); plt.imshow(patches[j]); plt.title('%.5f' % y_np[j]); plt.axis('off')
            # plt.tight_layout(); plt.show()
            optimizer.zero_grad()
            X, y = X.view(X.size(0) * X.size(1), X.size(2), X.size(3), X.size(4)).cuda(), y.flatten().cuda()
            y_ = net(X.cuda()).flatten()
            L = loss_fn(y_, y).mean()
            L.backward()
            optimizer.step()
            loss_history[-1]['train'].append(L.item())
        scheduler.step()

        net.eval()
        gt, pred = [], []
        with torch.no_grad():
            for i, (X, y) in enumerate(tqdm.tqdm(loader_valid, ascii=True, desc='evaluating %d/%d' % (ep + 1, args.epochs))):
                X, y = X.view(X.size(0) * X.size(1), X.size(2), X.size(3), X.size(4)).cuda(), y.flatten().cuda()
                y_ = net(X.cuda()).flatten().detach()
                L = loss_fn(y_, y).mean()
                loss_history[-1]['valid'].append(L.item())
        print('training loss = %.4f, validation loss = %.4f' % (np.array(loss_history[-1]['train']).mean(), np.array(loss_history[-1]['valid']).mean()))
        torch.save(net.state_dict(), 'iouestimator.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--image_batch_size', default=8, type=int)
    parser.add_argument('--bbox_batch_size', default=64, type=int)
    parser.add_argument('--lr', default=2e-3, type=int)
    parser.add_argument('--cocodir', type=str)
    args = parser.parse_args()
    print(args)

    assert 0 == args.bbox_batch_size % args.image_batch_size
    assert os.path.isdir(args.cocodir)
    train_eval(args)

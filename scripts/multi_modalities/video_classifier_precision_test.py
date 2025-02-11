#!python3

import numpy as np
import sklearn.utils
import skimage.io

import torch
import torch.utils.data as data
import torchvision
import torch.optim as optim
import torch.nn as nn

import os
import time
import copy
import tqdm
import glob
import argparse


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']


class VideoFramesTrain(data.Dataset):
    def __init__(self):
        super(VideoFramesTrain, self).__init__()
        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((448, 448), antialias=None),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.ifilelist, self.y = [], []
        for i, v in enumerate(video_id_list):
            files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', v, 'jpegs', '*.jpg'))))
            files = [files[j] for j in range(0, 2001, 100)]
            self.ifilelist = self.ifilelist + files
            self.y = self.y + [i] * len(files)
        self.ifilelist, self.y = map(np.array, [self.ifilelist, self.y])
        self.N = self.y.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        im = self.tf(torch.from_numpy(skimage.io.imread(self.ifilelist[i]).transpose(2, 0, 1)).float())
        im = im.view(3, 2, 224, 2, 224).transpose(2, 3).reshape(3, 4, 224, 224).transpose(0, 1)
        return im, torch.tensor([self.y[i] * 4, self.y[i] * 4 + 1, self.y[i] * 4 + 2, self.y[i] * 4 + 3]).long()


class VideoFramesValid(data.Dataset):
    def __init__(self, dtype=torch.float, repeat=1):
        super(VideoFramesValid, self).__init__()
        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((448, 448), antialias=None),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.dtype, self.repeat = dtype, repeat
        self.images, self.y = [], []
        for i, v in enumerate(video_id_list):
            ifilelist = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', v, 'jpegs', '*.jpg'))))
            ifilelist = [ifilelist[j] for j in range(-1, -2001, -400)]
            for f in ifilelist:
                im = self.tf(torch.from_numpy(skimage.io.imread(f).transpose(2, 0, 1)).float()).to(dtype=self.dtype)
                im = im.view(3, 2, 224, 2, 224).transpose(2, 3).reshape(3, 4, 224, 224).transpose(0, 1)
                self.images.append(im)
                self.y.append(torch.tensor([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3]).long())
        self.images = torch.cat(self.images, dim=0)
        self.y = torch.cat(self.y, dim=0)
        self.N = self.y.shape[0]

    def __len__(self):
        return self.N * self.repeat

    def __getitem__(self, i):
        i = i % self.N
        return self.images[i], self.y[i]


def train(args):
    np.random.seed(0)
    torch.manual_seed(0)
    assert(torch.cuda.is_available())
    print('GPU:', torch.cuda.get_device_name())

    net = torchvision.models.resnet101(pretrained=True)
    net.fc = nn.Linear(2048, 4 * len(video_id_list))
    net = net.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 2, gamma=0.1)
    loader_train = data.DataLoader(VideoFramesTrain(), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    loader_valid = data.DataLoader(VideoFramesValid(), batch_size=(args.batch_size * 4), shuffle=False, num_workers=1, pin_memory=False)

    for ep in range(0, args.epochs):
        net.train()
        for i, (X, y) in enumerate(tqdm.tqdm(loader_train, ascii=True, desc='training %d/%d' % (ep + 1, args.epochs))):
            X, y = X.view(-1, 3, 224, 224), y.view(-1)
            optimizer.zero_grad()
            y_ = net(X.cuda())
            L = loss_fn(y_, y.cuda())
            L.backward()
            optimizer.step()
        scheduler.step()

        net.eval()
        gt, pred = [], []
        with torch.no_grad():
            for i, (X, y) in enumerate(tqdm.tqdm(loader_valid, ascii=True, desc='evaluating %d/%d' % (ep + 1, args.epochs))):
                gt.append(y.numpy())
                pred.append(net(X.cuda()).detach().argmax(dim=1).cpu().numpy())
        gt, pred = map(np.concatenate, [gt, pred])
        print('accuracy: %.2f%%' % ((gt == pred).sum() / gt.shape[0] * 100))
    torch.save(net.state_dict(), 'r101_400_classes.pth')


def test(args):
    net = torchvision.models.resnet101()
    net.fc = nn.Linear(2048, 4 * len(video_id_list))
    net.load_state_dict(torch.load('r101_400_classes.pth'))
    net.eval()
    for _t in [torch.float, torch.half, torch.bfloat16]:
        net_t = copy.deepcopy(net).to(dtype=_t, device=torch.device('cuda'))
        net_t.eval()
        loader_valid = data.DataLoader(VideoFramesValid(dtype=_t, repeat=10), batch_size=(args.batch_size * 4), shuffle=False, num_workers=args.num_workers, pin_memory=False)
        gt, pred = [], []
        with torch.no_grad():
            for i, (X, y) in enumerate(tqdm.tqdm(loader_valid, ascii=True, desc='evaluating %s' % _t)):
                gt.append(y.numpy())
                pred.append(net_t(X.cuda()).detach().argmax(dim=1).cpu().numpy())
        gt, pred = map(np.concatenate, [gt, pred])
        print('accuracy: %.2f%%' % ((gt == pred).sum() / gt.shape[0] * 100))


class HarderMNIST(data.Dataset):
    def __init__(self, npz, split):
        super(HarderMNIST, self).__init__()
        assert split in ['train', 'valid']
        fp = np.load(npz)
        if split == 'train':
            self.X, self.y = fp['x_train'], fp['y_train']
        else:
            self.X, self.y = fp['x_test'], fp['y_test']
        fp.close()
        self.X, self.y = sklearn.utils.shuffle(self.X, self.y)
        self.X = self.X.astype(np.float32) / 255.0
        self.y = self.y.astype(np.int32)

    def __len__(self):
        return self.X.shape[0] - 5

    def __getitem__(self, i):
        Xi = self.X[i : i + 4].reshape(2, 2, 28, 28).transpose(0, 2, 1, 3).reshape(56, 56)
        Xi = torch.from_numpy(Xi).float()
        yi = torch.tensor(self.y[i] * 1000 + self.y[i + 1] * 100 + self.y[i + 2] * 10 + self.y[i + 3]).float() / 10000
        return Xi, yi


def testQ(args):
    np.random.seed(0)
    torch.manual_seed(0)

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(56 * 56, 10000), nn.ReLU(),
        nn.Linear(10000, 10000), nn.ReLU(),
        nn.Linear(10000, 10000), nn.ReLU(),
        nn.Linear(10000, 1),
    ).cuda()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 2, gamma=0.1)

    loader_train = data.DataLoader(HarderMNIST(args.mnist_npz, 'train'), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=False)
    loader_valid = data.DataLoader(HarderMNIST(args.mnist_npz, 'valid'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    net.train()
    for ep in range(0, args.epochs):
        for i, (X, y) in enumerate(tqdm.tqdm(loader_train, ascii=True, desc='training %d/%d' % (ep + 1, args.epochs))):
            optimizer.zero_grad()
            y_ = net(X.cuda()).view(-1)
            L = loss_fn(y_, y.cuda())
            L.backward()
            optimizer.step()
        scheduler.step()

    net = net.cpu()
    net.eval()

    gt, pred = [], []
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm.tqdm(loader_valid, ascii=True, desc='evaluating CPU')):
            gt.append(y.numpy())
            pred.append(net(X).view(-1).detach().cpu().numpy())
    gt, pred = map(np.concatenate, [gt, pred])
    print('MSE: %.4f' % np.absolute(gt - pred).mean())

    from torch.quantization import quantize_fx, quantize_dynamic

    net_Q = quantize_dynamic(copy.deepcopy(net), qconfig_spec={nn.Linear}, dtype=torch.qint8)
    gt, pred = [], []
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm.tqdm(loader_valid, ascii=True, desc='evaluating CPU quantized')):
            gt.append(y.numpy())
            pred.append(net(X).view(-1).detach().cpu().numpy())
    gt, pred = map(np.concatenate, [gt, pred])
    print('MSE: %.4f' % np.absolute(gt - pred).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--mnist_npz', type=str)
    args = parser.parse_args()
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.opt == 'train':
        train(args)
    elif args.opt == 'test':
        test(args)
    elif args.opt == 'testQ':
        testQ(args)

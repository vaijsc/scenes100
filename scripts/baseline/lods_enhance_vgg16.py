# coding=utf-8
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as transforms

from PIL import Image
import numpy as np
# from model.utils.config import cfg
import cv2
import matplotlib.pyplot as plt
import matplotlib
import os
import random


class enhance_base:
    def add_style(self, content, flag):
        if flag == 0:
            self.step += 1
        assert (len(content.size()) == 4)
        # if self.args.random_style:
        #     style=self.load_style_img(self.args,wh=(content.size(3),content.size(2)))
        # else:
        # if self.style_feats[flag][0].size() == content[0].size():
        # style = self.coral(self.style_feats[flag][0], content[0])
        style=self.style_feats[flag][0]
        # else:
        #     style=self.load_style_img(self.args,wh=(content.size(3),content.size(2)))
        with torch.no_grad():
            output = self.style_transfer(content[0], style, flag)
        for i in range(1, content.size(0)):
            # style = self.coral(self.style_feats[flag][0], content[i])
            with torch.no_grad():
                output = torch.cat((output, self.style_transfer(
                    content[i], style, flag)), 0)
        if flag==0:
            # clip 0-255
            output=(output.permute(0,2,3,1)+torch.from_numpy(self.pixel_means).float().cuda()).clamp(0,255)-torch.from_numpy(self.pixel_means).float().cuda()
            output=output.permute(0,3,1,2).contiguous()
            # if self.step%30==1:
            #     self.show(content,content=True)
            #     self.show(output)
        return output.detach()

    def __init__(self, encoders, decoders, fcs, video_id):
        assert len(encoders) == len(decoders)
        self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        self.target_size = 600
        self.encoders = encoders
        self.num = len(self.encoders)
        self.decoders = decoders
        self.fc1 = fcs[0]
        self.fc2 = fcs[1]
        for encoder in self.encoders:
            encoder.cuda()
        for decoder in self.decoders:
            decoder.cuda()
        self.fc1.cuda()
        self.fc2.cuda()
        for encoder in self.encoders:
            encoder.eval()
        for decoder in self.decoders:
            decoder.eval()
        self.fc1.eval()
        self.fc2.eval()
        self.style_feats = self.get_style_feats(video_id)

        # path = os.path.join(os.path.dirname(__file__), '..',self.args.log_dir,'noise')
        # if os.path.exists(path):
        #     import shutil
        #     shutil.rmtree(path)
        self.step = 0

    def get_style_feats(self, video_id):
        feats = []
        feats.append(self.load_style_img(video_id).unsqueeze(0))

        return feats

    def style_transfer(self, content, style, flag, alpha=1.0):
        assert (0.0 <= alpha <= 1.0)
        assert (len(content.size()) == 3)
        content = content.unsqueeze(0)
        style = style.unsqueeze(0)
        size=content.size()
        with torch.no_grad():
            for i in range(flag, self.num):
                content = self.encoders[i](content)
                style = self.encoders[i](style)
            feat = self.adaptive_instance_normalization(content, style, self.fc1, self.fc2)
            feat = feat * alpha + content * (1 - alpha)
            for i in range(self.num-flag):
                feat = self.decoders[i](feat)

        if feat.size()!=size:
            feat = torch.from_numpy(cv2.resize(feat[0].transpose(0, 1).transpose(1, 2).cpu().numpy(),(size[3],size[2]))).cuda().unsqueeze(0)
            feat = feat.permute(0,3,1,2)
        return feat

    def load_style_img(self, video_id):
        # if self.args.random_style:
        #     i = random.randint(0,len(self.imdb._image_index)-1)
        #     path = self.imdb.image_path_from_index(self.imdb._image_index[i])
        # else:
        #     print("random style is false! using style: "+args.style_path)
        #     path = args.style_path
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Intersections.LODS', 'Enhance', 'models', 'style_%s.png' % video_id))
        im = Image.open(path)
        im = im.convert('RGB')
        im = np.array(im)
        im = im[:, :, ::-1]
        im = im.astype(np.float32, copy=False)
        im -= self.pixel_means
        im_shape = im.shape

        im_size_min = np.min(im_shape[0:2])

        # if wh!=None:
        #     im = cv2.resize(im, wh,interpolation=cv2.INTER_LINEAR)
        # else:
        im_scale = float(self.target_size) / float(im_size_min)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2, 0, 1).contiguous()
        im = im.cuda()
        return im

    def coral(self, source, target):
        # assume both source and target are 3D array (C, H, W)
        # Note: flatten -> f

        # source style
        # target content
        source_f, source_f_mean, source_f_std = self._calc_feat_flatten_mean_std(
            source)
        source_f_norm = (source_f - source_f_mean.expand_as(
            source_f)) / source_f_std.expand_as(source_f)
        source_f_cov_eye = \
            torch.mm(source_f_norm, source_f_norm.t()) + \
            torch.eye(source.size(0)).cuda()

        target_f, target_f_mean, target_f_std = self._calc_feat_flatten_mean_std(
            target)
        target_f_norm = (target_f - target_f_mean.expand_as(
            target_f)) / target_f_std.expand_as(target_f)
        target_f_cov_eye = \
            torch.mm(target_f_norm, target_f_norm.t()) + \
            torch.eye(source.size(0)).cuda()

        source_f_norm_transfer = torch.mm(
            self._mat_sqrt(target_f_cov_eye),
            torch.mm(torch.inverse(self._mat_sqrt(source_f_cov_eye)),
                     source_f_norm)
        )

        source_f_transfer = source_f_norm_transfer * \
            target_f_std.expand_as(source_f_norm) + \
            target_f_mean.expand_as(source_f_norm)

        return source_f_transfer.view(source.size())

    def _mat_sqrt(self, x):
        U, D, V = torch.svd(x)
        return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

    def _calc_feat_flatten_mean_std(self, feat):
        # takes 3D feat (C, H, W), return mean and std of array within channels

        feat_flatten = feat.view(feat.size(0), -1)
        mean = feat_flatten.mean(dim=-1, keepdim=True)
        std = feat_flatten.std(dim=-1, keepdim=True)
        return feat_flatten, mean, std

    def adaptive_instance_normalization(self, content_feat, style_feat, fc1, fc2):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)

        mixed_style_mean = torch.cat((style_mean,content_mean),1).squeeze(2).squeeze(2)
        mixed_style_std = torch.cat((style_std,content_std),1).squeeze(2).squeeze(2)

        new_style_mean = (fc1(mixed_style_mean)).unsqueeze(2).unsqueeze(2)
        new_style_std = (fc2(mixed_style_std)).unsqueeze(2).unsqueeze(2)
        return normalized_feat * new_style_std.expand(size) + new_style_mean.expand(size)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def show(self, feat, content=False,save=True):
        for i in range(feat.size(0)):
            s = feat[i].transpose(
                0, 1).transpose(1, 2).cpu().numpy()
            s += self.pixel_means
            s = s[:, :, ::-1].astype(np.uint8)
            # Image.fromarray(s.astype(np.uint8)).show()
            if save:
                path = os.path.join(os.path.dirname(__file__), '..',self.args.log_dir,'noise')
                if not os.path.exists(path):
                    os.makedirs(path)
                if content:
                    matplotlib.image.imsave(os.path.join(path, 'step'+str(self.step)+'_real'+str(i)+'.jpg'), s)
                else:
                    matplotlib.image.imsave(os.path.join(path, 'step'+str(self.step)+'_'+str(i)+'.jpg'), s)
            else:
                plt.imshow(s)
                plt.show()


class enhance_vgg16(enhance_base):
    def __init__(self, video_id):
        self.video_id = video_id
        decoder = self.get_decoder()
        vgg = self.get_vgg()
        fcs = self.get_fcs()
        # print("using fcs...")
        vgg, decoder, fcs = self.load_param(vgg, decoder, fcs)
        self.encoders, self.decoders = self.splits(vgg, decoder)
        print('encoder/decoder loaded')
        enhance_base.__init__(self, self.encoders, self.decoders, fcs, self.video_id)

    def splits(self,vgg,decoder):
        encoders=[]
        decoders=[]
        encoders.append(nn.Sequential(*list(vgg._modules.values())[:2]))
        encoders.append(nn.Sequential(*list(vgg._modules.values())[2:7]))
        encoders.append(nn.Sequential(*list(vgg._modules.values())[7:12]))
        encoders.append(nn.Sequential(*list(vgg._modules.values())[12:]))
        decoders.append(nn.Sequential(*list(decoder._modules.values())[:7]))
        decoders.append(nn.Sequential(*list(decoder._modules.values())[7:12]))
        decoders.append(nn.Sequential(*list(decoder._modules.values())[12:17]))
        decoders.append(nn.Sequential(*list(decoder._modules.values())[17:]))
        return encoders,decoders

    def get_fcs(self):
        fc1 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(inplace=True),nn.Linear(512,512))
        fc2 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(inplace=True),nn.Linear(512,512))
        return [fc1,fc2]

    def get_vgg(self):
        vgg = models.vgg16()
        vgg = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        vgg[4].ceil_mode = True
        vgg[9].ceil_mode = True
        vgg[16].ceil_mode = True
        return vgg

    def get_decoder(self):
        decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
        )
        return decoder

    def load_param(self, vgg, decoder, fcs):
        model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Intersections.LODS', 'Enhance', 'models'))
        for param in vgg.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        for i in range(len(fcs)):
            for param in fcs[i].parameters():
                param.requires_grad = False
        decoder.load_state_dict(torch.load(os.path.join(model_dir, 'decoder_%s_iter_50000.pth' % self.video_id)))
        vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16_ori.pth'))['model'])
        vgg = nn.Sequential(*list(vgg.children())[:19])
        fcs[0].load_state_dict(torch.load(os.path.join(model_dir, 'fc1_%s_iter_50000.pth' % self.video_id)))
        fcs[1].load_state_dict(torch.load(os.path.join(model_dir, 'fc2_%s_iter_50000.pth' % self.video_id)))
        # print("loaded encoder: "+args.encoder_path)
        # print("loaded decoder: "+args.decoder_path)
        # print("loaded fc1: "+args.fc1)
        # print("loaded fc2: "+args.fc2)
        # if args.random_style:
        #     print("random style is True")
        # else:
        #     print("random style is False")
        return vgg, decoder, fcs


def generate_style_images():
    import glob
    import tqdm
    import random
    import skimage.io
    model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Intersections.LODS', 'Enhance', 'models'))
    for video_id in tqdm.tqdm(['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179'], ascii=True):
        images = glob.glob(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', video_id, 'jpegs', '*.jpg'))
        random.shuffle(images)
        images = images[:150]
        images = list(map(skimage.io.imread, images))
        images = np.stack(images, axis=0).astype(np.float16)
        images = images.mean(axis=0).astype(np.uint8)
        skimage.io.imsave(os.path.join(model_dir, 'style_%s.png' % video_id), images)


if __name__ == '__main__':
    generate_style_images()

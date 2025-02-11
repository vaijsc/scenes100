#!python3

import numpy as np
import skimage.io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision

'''
https://arxiv.org/pdf/1604.03901.pdf
https://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_Monocular_Relative_Depth_CVPR_2018_paper.pdf
https://arxiv.org/pdf/2104.06456.pdf
'''


class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Identity()
        # self.bn2 = nn.Identity()
        if in_channels != out_channels:
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, (1, 1), stride=1, padding=0, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1 = nn.Identity()
            self.bn_1x1 = nn.Identity()

    def forward(self, X):
        out_1 = self.relu(self.bn1(self.conv1(X)))
        out_2 = self.bn2(self.conv2(out_1))
        r = self.relu(out_2 + self.bn_1x1(self.conv_1x1(X)))
        return r


class DepthEstimator(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthEstimator, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.upsample2x = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        self.conv_1x1_4 = nn.Sequential(
            nn.Conv2d(2048, 1024, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.fusion4 = BasicResidualBlock(1024, 1024)
        self.fusion3 = BasicResidualBlock(1024, 512)
        self.fusion2 = BasicResidualBlock(512, 256)
        self.fusion1 = BasicResidualBlock(256, 64)
        self.fusion0 = BasicResidualBlock(64, 64)
        self.conv_output = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
        )

    def _set_backbone_grad(self, flag):
        for p in self.backbone.parameters():
            p.requires_grad = flag
    def backbone_freeze(self):
        print('freeze ResNet backbone')
        self._set_backbone_grad(False)
    def backbone_unfreeze(self):
        print('unfreeze ResNet backbone')
        self._set_backbone_grad(True)

    def forward(self, X):
        _, C, H, W = X.size()
        X_scale = nn.functional.interpolate(X, size=448, mode='bilinear', align_corners=True) # 3, W, W

        out_0 = self.backbone.conv1(X_scale)
        out_0 = self.backbone.bn1(out_0)
        out_0 = self.backbone.relu(out_0)
        out_0 = self.backbone.maxpool(out_0) # 64, /4, /4

        out_1 = self.backbone.layer1(out_0) # 256, /4, /4
        out_2 = self.backbone.layer2(out_1) # 512, /8, /8
        out_3 = self.backbone.layer3(out_2) # 1024, /16, /16
        out_4 = self.backbone.layer4(out_3) # 2048, /32, /32

        rc_4 = self.upsample2x(self.fusion4(self.conv_1x1_4(out_4))) # 1024, /16, /16
        rc_3 = self.upsample2x(self.fusion3(rc_4 + out_3)) # 512, /8, /8
        rc_2 = self.upsample2x(self.fusion2(rc_3 + out_2)) # 256, /4, /4
        rc_1 = self.fusion1(rc_2 + out_1) # 64, /4, /4
        rc_0 = self.upsample2x(self.fusion0(rc_1 + out_0)) # 64, /2, /2
        rc = self.upsample2x(self.conv_output(rc_0)) # 1, W, W
        rc_HW = nn.functional.interpolate(rc, size=(H, W), mode='bilinear', align_corners=True)

        # outs = [X, X_scale, out_0, out_1, out_2, out_3, out_4, rc_4, rc_3, rc_2, rc_1, rc_0, rc, rc_HW]
        # names = ['X', 'X_scale', 'out_0', 'out_1', 'out_2', 'out_3', 'out_4', 'rc_4', 'rc_3', 'rc_2', 'rc_1', 'rc_0', 'rc', 'rc_HW']
        # plt.figure()
        # for i in range(0, len(outs)):
        #     plt.subplot(2, 7, i + 1)
        #     im = outs[i].detach().cpu().numpy()[0].transpose(1, 2, 0)
        #     im -= im.min()
        #     im /= im.max()
        #     plt.imshow(im[:, :, :3])
        #     plt.title('%s %s' % (names[i], im.shape))
        # plt.show()
        # return rc, rc_HW
        return rc_HW


if __name__ == '__main__':
    m = DepthEstimator()
    norm = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x = skimage.io.imread('test.jpg').transpose(2, 0, 1)
    x = torch.from_numpy(x / 255.0).unsqueeze(0).float()
    x = norm(x)
    o = m(x)
    print(x.size(), o[0].size(), o[1].size())

    # torch.save(m.backbone.state_dict(), 'backhone.pth')
    # torch.save(m.state_dict(), 'model.pth')

    # m = BasicResidualBlock(128, 256)
    # x = torch.randn(1, 128, 16, 16)
    # print(x.size(), m(x).size())

    pass

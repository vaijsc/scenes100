#!python3

# https://github.com/tchittesh/lzu/blob/main/lzu/fixed_grid.py
# https://github.com/tchittesh/lzu/blob/main/lzu/invert_grid.py

import os
import pickle
import math
import skimage
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

# from mmdet3d.models.builder import MODELS
# GRID_GENERATORS = MODELS


# def build_grid_generator(cfg):
#     """Build view transformer."""
#     return GRID_GENERATORS.build(cfg)


def make1DGaussian(size, fwhm=3, center=None):
    """ Make a 1D gaussian kernel.

    size is the length of the kernel,
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, dtype=float)

    if center is None:
        center = size // 2

    return np.exp(-4*np.log(2) * (x-center)**2 / fwhm**2)


def make2DGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


class RecasensSaliencyToGridMixin(object):
    """Grid generator based on 'Learning to Zoom: a Saliency-Based Sampling \
    Layer for Neural Networks' [https://arxiv.org/pdf/1809.03355.pdf]."""

    def __init__(self, output_shape, grid_shape=(31, 51), separable=True,
                 attraction_fwhm=13, anti_crop=True, **kwargs):
        super(RecasensSaliencyToGridMixin, self).__init__()
        self.output_shape = output_shape
        self.output_height, self.output_width = output_shape
        self.grid_shape = grid_shape
        self.padding_size = min(self.grid_shape)-1
        self.total_shape = tuple(
            dim+2*self.padding_size
            for dim in self.grid_shape
        )
        self.padding_mode = 'reflect' if anti_crop else 'replicate'
        self.separable = separable

        if self.separable:
            self.filter = make1DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter).unsqueeze(0) \
                                                        .unsqueeze(0).cuda()

            self.P_basis_x = torch.zeros(self.total_shape[1])
            for i in range(self.total_shape[1]):
                self.P_basis_x[i] = \
                    (i-self.padding_size)/(self.grid_shape[1]-1.0)
            self.P_basis_y = torch.zeros(self.total_shape[0])
            for i in range(self.total_shape[0]):
                self.P_basis_y[i] = \
                    (i-self.padding_size)/(self.grid_shape[0]-1.0)
        else:
            self.filter = make2DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter) \
                               .unsqueeze(0).unsqueeze(0).cuda()

            self.P_basis = torch.zeros(2, *self.total_shape)
            for k in range(2):
                for i in range(self.total_shape[0]):
                    for j in range(self.total_shape[1]):
                        self.P_basis[k, i, j] = k*(i-self.padding_size)/(self.grid_shape[0]-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_shape[1]-1.0)  # noqa: E501

    def separable_saliency_to_grid(self, imgs, x_saliency,
                                   y_saliency, device):
        assert self.separable
        x_saliency = F.pad(x_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)
        y_saliency = F.pad(y_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)

        N = imgs.shape[0]
        P_x = torch.zeros(1, 1, self.total_shape[1], device=device)
        P_x[0, 0, :] = self.P_basis_x
        P_x = P_x.expand(N, 1, self.total_shape[1])
        P_y = torch.zeros(1, 1, self.total_shape[0], device=device)
        P_y[0, 0, :] = self.P_basis_y
        P_y = P_y.expand(N, 1, self.total_shape[0])

        weights = F.conv1d(x_saliency, self.filter)
        weighted_offsets = torch.mul(P_x, x_saliency)
        weighted_offsets = F.conv1d(weighted_offsets, self.filter)
        xgrid = weighted_offsets/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, 1, self.grid_shape[1])
        xgrid = xgrid.expand(-1, 1, *self.grid_shape)

        weights = F.conv1d(y_saliency, self.filter)
        weighted_offsets = F.conv1d(torch.mul(P_y, y_saliency), self.filter)
        ygrid = weighted_offsets/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, self.grid_shape[0], 1)
        ygrid = ygrid.expand(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        upsampled_grid = F.interpolate(grid, size=self.output_shape,
                                       mode='bilinear', align_corners=True)
        return upsampled_grid.permute(0, 2, 3, 1), grid.permute(0, 2, 3, 1)

    def nonseparable_saliency_to_grid(self, imgs, saliency, device):
        assert not self.separable
        p = self.padding_size
        saliency = F.pad(saliency, (p, p, p, p), mode=self.padding_mode)

        N = imgs.shape[0]
        P = torch.zeros(1, 2, *self.total_shape, device=device)
        P[0, :, :, :] = self.P_basis
        P = P.expand(N, 2, *self.total_shape)

        saliency_cat = torch.cat((saliency, saliency), 1)
        weights = F.conv2d(saliency, self.filter)
        weighted_offsets = torch.mul(P, saliency_cat) \
                                .view(-1, 1, *self.total_shape)
        weighted_offsets = F.conv2d(weighted_offsets, self.filter) \
                            .view(-1, 2, *self.grid_shape)

        weighted_offsets_x = weighted_offsets[:, 0, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        xgrid = weighted_offsets_x/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, *self.grid_shape)

        weighted_offsets_y = weighted_offsets[:, 1, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        ygrid = weighted_offsets_y/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        upsampled_grid = F.interpolate(grid, size=self.output_shape,
                                       mode='bilinear', align_corners=True)
        return upsampled_grid.permute(0, 2, 3, 1), grid.permute(0, 2, 3, 1)


# @GRID_GENERATORS.register_module()
class FixedGrid(nn.Module, RecasensSaliencyToGridMixin):
    """Grid generator that uses a fixed saliency map -- KDE SD"""

    def __init__(self, saliency_file, **kwargs):
        super(FixedGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.saliency = pickle.load(open(saliency_file, 'rb')).cuda()

        if self.separable:
            x_saliency = self.saliency.sum(dim=2)
            y_saliency = self.saliency.sum(dim=3)
            self.upsampled_grid, self.grid = self.separable_saliency_to_grid(
                torch.zeros(1), x_saliency, y_saliency, torch.device('cuda'))
        else:
            self.upsampled_grid, self.grid = (
                self.nonseparable_saliency_to_grid(
                    torch.zeros(1), self.saliency, torch.device('cuda'))
            )

    def forward(self, imgs, img_metas, **kwargs):
        B = imgs.shape[0]
        upsampled_grid = self.upsampled_grid.expand(B, -1, -1, -1)
        grid = self.grid.expand(B, -1, -1, -1)

        # Uncomment to visualize saliency map
        # h, w, _ = img_metas[0]['pad_shape']
        # show_saliency = F.interpolate(self.saliency, size=(h, w),
        #                                 mode='bilinear', align_corners=True)
        # show_saliency = 255*(show_saliency/show_saliency.max())
        # show_saliency = show_saliency.expand(
        #     show_saliency.size(0), 3, h, w)
        # vis_batched_imgs(vis_options['saliency'], show_saliency,
        #                     img_metas, denorm=False)
        # vis_batched_imgs(vis_options['saliency']+'_no_box', show_saliency,
        #                     img_metas, bboxes=None, denorm=False)

        return upsampled_grid, grid


class LZUTransform(nn.Module, RecasensSaliencyToGridMixin):
    def __init__(self, **kwargs):
        super(LZUTransform, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)

        # pre trained
        # saliency = pickle.load(open(os.path.join(os.path.dirname(__file__), 'lzu_saliency.pkl'), 'rb')) # 1 x 1 x grid-H x grid-W

        # uniform
        saliency = torch.ones(size=self.grid_shape, dtype=torch.float32)
        saliency += torch.randn(size=saliency.size()) * 0.01
        saliency = saliency / saliency.sum()
        saliency = saliency.unsqueeze(0).unsqueeze(0)

        assert saliency.size(0) == saliency.size(1) == 1
        saliency_logit = torch.log(saliency)
        _c = -1 * saliency_logit.sum() / np.prod(saliency_logit.size())
        saliency_logit = saliency_logit + _c
        self.saliency_logit = torch.nn.Parameter(saliency_logit.cuda())

    def load_saliency(self, saliency: np.ndarray):
        assert saliency.shape[0] == 27 and saliency.shape[1] == 48, saliency.shape
        saliency = torch.from_numpy(saliency).float().unsqueeze(0).unsqueeze(0)
        saliency_logit = torch.log(saliency)
        _c = -1 * saliency_logit.sum() / np.prod(saliency_logit.size())
        saliency_logit = saliency_logit + _c
        self.load_state_dict({'saliency_logit': saliency_logit.cuda()})

    def get_saliency(self):
        saliency = F.softmax(self.saliency_logit.flatten(), dim=0)
        return saliency.view(*self.saliency_logit.size())

    def forward(self, imgs):
        raise NotImplementedError

    def zoom(self, imgs):
        assert self.separable
        device = self.saliency_logit.device
        saliency = self.get_saliency()
        x_saliency = saliency.sum(dim=2)
        y_saliency = saliency.sum(dim=3)
        x_saliency = F.pad(x_saliency, (self.padding_size, self.padding_size), mode=self.padding_mode)
        y_saliency = F.pad(y_saliency, (self.padding_size, self.padding_size), mode=self.padding_mode)

        N, _, H, W = imgs.size()
        P_x = torch.zeros(1, 1, self.total_shape[1], device=device)
        P_x[0, 0, :] = self.P_basis_x
        P_x = P_x.expand(1, 1, self.total_shape[1])
        P_y = torch.zeros(1, 1, self.total_shape[0], device=device)
        P_y[0, 0, :] = self.P_basis_y
        P_y = P_y.expand(1, 1, self.total_shape[0])

        weights = F.conv1d(x_saliency, self.filter)
        weighted_offsets = torch.mul(P_x, x_saliency)
        weighted_offsets = F.conv1d(weighted_offsets, self.filter)
        xgrid = weighted_offsets / weights
        xgrid = torch.clamp(xgrid * 2 - 1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, 1, self.grid_shape[1])
        xgrid = xgrid.expand(-1, 1, *self.grid_shape)

        weights = F.conv1d(y_saliency, self.filter)
        weighted_offsets = F.conv1d(torch.mul(P_y, y_saliency), self.filter)
        ygrid = weighted_offsets / weights
        ygrid = torch.clamp(ygrid * 2 - 1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, self.grid_shape[0], 1)
        ygrid = ygrid.expand(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        upsampled_grid = F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=True)
        imgs_zoom = F.grid_sample(imgs, upsampled_grid.permute(0, 2, 3, 1).expand(N, -1, -1, -1), mode='bilinear', align_corners=True, padding_mode='zeros')
        return imgs_zoom, grid.permute(0, 2, 3, 1)

    def unzoom(self, imgs, grid):
        device = self.saliency_logit.device
        N, _, H, W = imgs.size()
        B, grid_H, grid_W, _ = grid.size()
        assert B == 1 # assume grid for all images in batch is the same

        grid = torch.clone(grid)
        eps = 1e-8
        grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * (W - 1)
        grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * (H - 1)
        # grid now ranges from 0 to ([H or W] - 1)
        # TODO: implement batch operations
        inverse_grid = 2 * max(H, W) * torch.ones([1, H, W, 2], dtype=torch.float32, device=device)

        # each of these is ((grid_H - 1)*(grid_W - 1)) x 2
        p00 = grid[0,  :-1,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p10 = grid[0, 1:  ,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p01 = grid[0,  :-1,  1:  , :].contiguous().view(-1, 2)  # noqa: 203

        ref = torch.floor(p00).to(torch.int)
        v00 = p00 - ref
        v10 = p10 - ref
        v01 = p01 - ref
        vx = p01[:, 0] - p00[:, 0]
        vy = p10[:, 1] - p00[:, 1]

        min_x = int(math.floor(v00[:, 0].min() - eps))
        max_x = int(math.ceil(v01[:, 0].max() + eps))
        min_y = int(math.floor(v00[:, 1].min() - eps))
        max_y = int(math.ceil(v10[:, 1].max() + eps))

        pts = torch.cartesian_prod(
            torch.arange(min_x, max_x + 1, device=device),
            torch.arange(min_y, max_y + 1, device=device),
        ).T  # 2 x (x_range*y_range)

        unwarped_x = (pts[0].unsqueeze(0) - v00[:, 0].unsqueeze(1)) / vx.unsqueeze(1)  # noqa: E501
        unwarped_y = (pts[1].unsqueeze(0) - v00[:, 1].unsqueeze(1)) / vy.unsqueeze(1)  # noqa: E501
        unwarped_pts = torch.stack((unwarped_y, unwarped_x), dim=0)  # noqa: E501, has shape 2 x ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)

        good_indices = torch.logical_and(
            torch.logical_and(-eps <= unwarped_pts[0], unwarped_pts[0] <= 1 + eps),
            torch.logical_and(-eps <= unwarped_pts[1], unwarped_pts[1] <= 1 + eps),
        )  # ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        nonzero_good_indices = good_indices.nonzero()
        inverse_j = pts[0, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 0]  # noqa: E501
        inverse_i = pts[1, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 1]  # noqa: E501

        # TODO: is replacing this with reshape operations on good_indices faster? # noqa: E501
        j = nonzero_good_indices[:, 0] % (grid_W - 1)
        i = nonzero_good_indices[:, 0] // (grid_W - 1)
        grid_mappings = torch.stack(
            (j + unwarped_pts[1, good_indices], i + unwarped_pts[0, good_indices]),  # noqa: E501
            dim=1
        )
        in_bounds = torch.logical_and(
            torch.logical_and(0 <= inverse_i, inverse_i < H),
            torch.logical_and(0 <= inverse_j, inverse_j < W),
        )
        inverse_grid[0, inverse_i[in_bounds], inverse_j[in_bounds], :] = grid_mappings[in_bounds, :]  # noqa: E501

        inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W - 1) * 2.0 - 1.0  # noqa: E501
        inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H - 1) * 2.0 - 1.0  # noqa: E501
        imgs_unzoom = F.grid_sample(imgs, inverse_grid.expand(N, -1, -1, -1), mode='bilinear', align_corners=True, padding_mode='zeros')
        return imgs_unzoom


if __name__ == '__main__':
    net = LZUTransform(
        output_shape=(1080, 1920),
        grid_shape=(27, 48),
        separable=True,
        attraction_fwhm=10,
        anti_crop=True
    )
    net.load_saliency(np.load(os.path.join(os.path.dirname(__file__), 'lzu_saliency', '001.npy')))
    for p in net.parameters():
        print('parameter:', p.size())
    print(net.get_saliency().sum())
    print(net.get_saliency()[0, 0, :, 0])

    x = skimage.io.imread(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', '001', 'unmasked', '00164159.jpg'))
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().expand(5, -1, -1, -1) / 255.0
    x_zoom, grid = net.zoom(x.cuda())
    print(x.size(), x_zoom.size(), grid.size())
    x_unzoom = net.unzoom(x_zoom, grid)
    x_zoom.mean().backward(retain_graph=True)
    print('gradient:', net.saliency_logit.grad[0, 0, 1])
    net.saliency_logit.grad = None
    x_unzoom.mean().backward(retain_graph=True)
    print('gradient:', net.saliency_logit.grad[0, 0, 1])

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(net.get_saliency()[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 2)
    im_input = x[0].numpy().transpose(1, 2, 0)
    plt.imshow(im_input)
    plt.subplot(2, 2, 3)
    im_zoom = x_zoom[0].detach().cpu().numpy().transpose(1, 2, 0)
    im_zoom[:, :, 2] = im_input[:, :, 2]
    plt.imshow(im_zoom)
    plt.subplot(2, 2, 4)
    im_unzoom = x_unzoom[0].detach().cpu().numpy().transpose(1, 2, 0)
    im_unzoom[:, :, 2] = im_input[:, :, 2]
    plt.imshow(im_unzoom)
    plt.show()

import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from math import exp

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class PerceptualLosses(nn.Module):
    def __init__(self, gpu_ids):
        super(PerceptualLosses, self).__init__()
        self.vgg = Vgg19().cuda()
        self.san = SketchANet().cuda()
        self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        x_san, y_san = self.san(x), self.san(y)
        vgg_content_loss = 0
        for i in range(len(x_vgg)):
            vgg_content_loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        vgg_style_loss = 0
        for i in range(len(x_vgg)):
            vgg_style_loss += self.weights[i] * self.criterion(gram_matrix(x_vgg[i]), gram_matrix(y_vgg[i].detach()))

        san_content_loss = 0
        for i in range(len(x_san)):
            san_content_loss += self.weights[i] * self.criterion(x_san[i], y_san[i].detach())

        san_style_loss = 0
        for i in range(len(x_san)):
            san_style_loss += self.weights[i] * self.criterion(gram_matrix(x_san[i]), gram_matrix(y_san[i].detach()))

        loss = vgg_content_loss + vgg_style_loss + san_content_loss + san_style_loss
        return loss

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class SketchANetLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(SketchANetLoss, self).__init__()
        self.SketchANet().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_sketchanet, y_sketchanet = self.sketchanet(x), self.sketchanet(y)
        loss = 0
        for i in range(len(x_sketchanet)):
            loss += self.weights * self.criterion(x_sketchanet[i], y_sketchanet[i].detach())
        return loss

from .sketchanet import sketchanetmodel
class SketchANet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(SketchANet, self).__init__()
        sketchanet_pretrained_features = sketchanetmodel(pretrained=True).base.conv
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), sketchanet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), sketchanet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), sketchanet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), sketchanet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), sketchanet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()
#
# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window
#
# def _ssim(img1, img2, window, window_size, channel, size_average = True):
#     mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
#     mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1*mu2
#
#     sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
#
#     C1 = 0.01**2
#     C2 = 0.03**2
#
#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
#
#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
#
#
# class SSIM(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)
#
#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()
#
#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
#
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
#
#             self.window = window
#             self.channel = channel
#
#         return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
#
#
# def ssim(img1, img2, window_size=11, size_average=True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
#
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#
#     return _ssim(img1, img2, window, window_size, channel, size_average)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, full=False):
    padd = 0

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True, full=False):
    (_, channel, height, width) = img1.size()

    real_size = min(window_size, height, width)
    window = create_window(real_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, real_size, channel, size_average, full=full)


def msssim(img1, img2, window_size=11, size_average=True):
    # TODO: fix NAN results
    if img1.size() != img2.size():
        raise RuntimeError('Input images must have the same shape (%s vs. %s).' %
                           (img1.size(), img2.size()))
    if len(img1.size()) != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           len(img1.size()))

#    if type(img1) is not Variable or type(img2) is not Variable:
#        raise RuntimeError('Input images must be Variables, not %s' %
#                           img1.__class__.__name__)

    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    if img1.is_cuda:
        weights = weights.cuda(img1.get_device())

    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True)
        mssim.append(sim.unsqueeze(0))
        mcs.append(cs.unsqueeze(0))

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    mssim = torch.cat(mssim)
    mcs = torch.cat(mcs)
    return (torch.prod(mcs[0:levels - 1] ** weights[0:levels - 1]) *
            (mssim[levels - 1] ** weights[levels - 1]))


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

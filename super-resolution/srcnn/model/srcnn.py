import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_networks import *
import skimage as sk
import math

import pytorch_ssim as ps
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

class Net(nn.Module):
    def __init__(self, params=None):
        super(Net, self).__init__()
        self.num_channels = params.num_channels if params is not None else 64
        self.dropout_rate = params.dropout_rate if params is not None else 0.0
        
        self.layers = torch.nn.Sequential(
            ConvBlock(3, self.num_channels, 9, 1, 4, norm=None),  # 144*144*64 # conv->batchnorm->activation
            ConvBlock(self.num_channels, self.num_channels // 2, 1, 1, 0, norm=None),  # 144*144*32
            ConvBlock(self.num_channels // 2, 3, 5, 1, 2, activation=None, norm=None)  # 144*144*1
        )

    def forward(self, s):
        out = self.layers(s)
        return out

def loss_fn(outputs, labels):
    N, C, H, W = outputs.shape
        
    mse_loss = torch.sum((outputs - labels) ** 2) / N / C   # each photo, each channel
    mse_loss *= 255 * 255
    mse_loss /= H * W  
    # average loss on each pixel(0-255)
    return mse_loss

def accuracy(outputs, labels):
    N, _, _, _ = outputs.shape
    psnr = 0
    for i in range(N):
        psnr += compare_psnr(labels[i],outputs[i])
    return psnr / N

def ssim(outputs, labels):
    N, _, _, _ = outputs.shape
    ssim = 0
    for i in range(N):
        ssim += compare_ssim(labels[i], outputs[i], win_size=3, multichannel=True, data_range=1.0)

    return ssim / N    

metrics = {
    'PSNR': accuracy,
    'SSIM': ssim,
}


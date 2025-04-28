import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .resnet import ResNet

class SpatioTSEncoder(nn.Module):
    def __init__(self, input_dims, hist_length, hidden_dims, output_dims,
                 kernel_size, conv1d_kernel_size, num_layers, device = 0):

        super().__init__()

        self.feature_extractor = ResNet()
        
        # device to use
        self.device = device


    def forward(self, x):  # x: B x T x R x H x W

        # B - batch size. We will set it to 1 to increase number of regions for
        #     contrasting
        # T - history length
        # R - overall number of regions
        # H , W - shape of region, which name is in "dataset_name"
        B, T, R, H, W = x.shape

        # ResNet encoder
        # [B, T, R, H, W] -> [R, T, C]
        h = self.feature_extractor(x)

        return h
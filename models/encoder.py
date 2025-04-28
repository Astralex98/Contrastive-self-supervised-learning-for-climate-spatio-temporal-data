import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .convlstm import ConvLSTM
# from .dilated_conv import DilatedConvEncoder

class SpatioTSEncoder(nn.Module):
    def __init__(self, input_dims, hist_length, hidden_dims, cell_output_dims,
                 region_output_dims, kernel_size, conv1d_kernel_size, num_layers, device = 0):

        super().__init__()

        # overall number of regions (R)
        self.R = input_dims
        
        # history length for self-supervised training (T)
        self.T = hist_length
        
        # Hidden dimension of the encoder
        self.hidden_dims = hidden_dims
        
        # Output dimension of the encoder (the representation dimension of a cell)
        self.cell_output_dims = cell_output_dims
        
        # The representation dimension of the region
        self.region_output_dims = region_output_dims
        
        self.conv1d_kernel_size = conv1d_kernel_size

        # create a list of hidden dims for ConvLSTM
        hid_dims_list = [self.hidden_dims for _ in range(num_layers - 1)]
        hid_dims_list.append(self.cell_output_dims)
        
        self.feature_extractor = ConvLSTM(input_dim = self.R,
                                          hidden_dim = hid_dims_list,
                                          kernel_size = kernel_size,
                                          num_layers = num_layers,
                                          batch_first=True,
                                          return_all_layers=False)
        
        self.fc = torch.nn.Linear(self.cell_output_dims, self.region_output_dims)
        
        self.m = torch.nn.Conv1d(1,  self.R, kernel_size = self.conv1d_kernel_size, padding='same')
        self.n = torch.nn.Conv1d(1, self.T, kernel_size = self.conv1d_kernel_size, padding='same')
        
        # device to use
        self.device = device


    def forward(self, x):  # x: B x T x R x H x W

        # B - batch size. We will set it to 1 to increase number of regions for
        #     contrasting
        # T - history length
        # R - overall number of regions
        # H , W - shape of region, which name is in "dataset_name"
        B, T, R, H, W = x.shape

        # conv encoder
        _ , last_states = self.feature_extractor(x)


        # h: B x С(representation_dim) x H x W
        h = last_states[0][0]

        # [1, C, H, W] -> [1, C]
        # [1, C, H, W] -> [1, C, H*W] -> (take average by the last dimension) [1, C]
        h = torch.reshape(h, (h.shape[0], h.shape[1], -1)).mean(dim=-1)
        
        # [1, C] -> [1, C*]
        # C* - representation of the region (C* = self.region_output_dims)
        h = self.fc(h)

        # [1, C*] -> [1, R, C*]
        h = torch.reshape(h, (h.shape[0], 1, -1))
        h = self.m(h)

        # [1, R, C*] -> [R, T, C*]
        h = torch.reshape(h, (h.shape[1], 1, -1))
        h = self.n(h)
        # h: [R, T, C*]

        return h
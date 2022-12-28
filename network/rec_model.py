"""
paper: T-Net: Deep Stacked Scale-Iteration Network for Image Dehazing
file: rec_model.py
about: recurrent model for Stack T-Net
author: Lirong Zheng
date: 01/01/21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TNet

class RecModel(nn.Module):
    def __init__(self, recurrent_iter=2, height=4, width=6, depth_rate=16, num_dense_layer=4, growth_rate=16):
        super(RecModel, self).__init__()
        self.iteration = recurrent_iter
        self.GridNet = TNet(height, width, depth_rate, num_dense_layer, growth_rate)

    def forward(self, input):
        x = input
        x_list = []
        for i in range(self.iteration):
            x = torch.cat((x, input), 1)
            x = self.GridNet(x)
            x_list.append(x)
        return x, x_list
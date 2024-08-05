# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch
import torch.nn as nn


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)


class MlpRandomNoise(nn.Module):
    def __init__(self, c_in, mlp_channels=None, ret_before_act=False, without_norm=False, mean=0.0, std=0.1):
        super().__init__()
        self.mlp = build_mlps(c_in, mlp_channels, ret_before_act, without_norm)
        self.mean = mean
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return self.mlp(x + noise)
    

def build_mlps_with_noise(c_in, mlp_channels=None, ret_before_act=False, without_norm=False, mean=0.0, std=0.1):
    return MlpRandomNoise(c_in, mlp_channels, ret_before_act, without_norm, mean, std)


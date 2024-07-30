import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layer import Transformer
from einops import rearrange

# class TransformerModule(nn.Module):
#     def __init__(self, d_model, nhead, num_layers, dim_feedforward):
#         super(TransformerModule, self).__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
#
#     def forward(self, x):
#         return self.encoder(x)

class ActorNetwork(nn.Module):
    def __init__(self, action_size):
        super(ActorNetwork, self).__init__()
        # self.fc1 = nn.Linear(7, 32)
        # self.movement_transformer = TransformerModule(32, 8, 4, 32)
        # self.pooling = nn.AdaptiveAvgPool1d(32)
        # self.time_transformer = TransformerModule(32, 8, 4, 32)
        # self.fc2 = nn.Linear(32, action_size)
        self.fc1 = nn.Linear(12, 64)
        self.transformer1 = Transformer(dim=64, depth=2, heads=8, dim_head=128, mlp_dim=128, dropout=0.1)
        self.fc2 = nn.Linear(64, action_size)
        self.pooling = nn.AdaptiveAvgPool2d((1, 64))

    def forward(self, x):
        x = x['local']
        if x.ndim == 5:
            ea = list(x.shape[:2])
            x = rearrange(x, 'e a ... -> (e a) ...')

        elif x.ndim == 4:
            ea = list([x.shape[0]])
        else:
            raise 'unexpected ndim'

        time, movement = x.shape[-3:-1]
        x = x[..., 0]  # 只取占有率

        x = self.fc1(x)
        x = self.transformer1(x)
        x = self.pooling(x)
        x = rearrange(x, 'ea 1 d -> ea d')
        x = self.fc2(x)

        return x.view(*ea, -1)
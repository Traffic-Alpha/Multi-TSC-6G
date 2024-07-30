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
        self.fc1 = nn.Linear(84, 64)
        self.transformer1 = Transformer(dim=64, depth=2, heads=8, dim_head=128, mlp_dim=128, dropout=0.1)
        self.fc3 = nn.Linear(64, action_size)
        self.pooling = nn.AdaptiveAvgPool2d((1, 64))

    def forward(self, x):
        x = x['local']
        if x.ndim == 5:
            ea = list(x.shape[:2])
            x = rearrange(x, 'e a ... -> (e a) ...')
        elif x.ndim == 4:
            ea = list(x.shape[0])
        else:
            raise 'unexpected ndim'

        time, movement, feature = x.shape[-3:]

        x = rearrange(x, pattern='... m f -> ... (m f)')
        x = self.fc1(x)
        x = self.transformer1(x)
        x = self.pooling(x)
        x = rearrange(x, 'ea 1 d -> ea d')
        x = self.fc3(x)

        return x.view(*ea, -1)
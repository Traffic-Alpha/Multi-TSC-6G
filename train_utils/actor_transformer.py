import torch
import torch.nn as nn
import torch.nn.functional as F
from .projector import Projector

class TransformerModule(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerModule, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        return self.encoder(x)

class ActorTransformer(nn.Module):
    def __init__(self):
        super(ActorTransformer, self).__init__()
        self.fc1 = nn.Linear(7, 32)
        self.movement_transformer = TransformerModule(32, 8, 4, 32)
        self.pooling = nn.AdaptiveAvgPool1d(32)
        self.time_transformer = TransformerModule(32, 8, 4, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        n_envs, batch = list(x.shape[:-3]) # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x.shape[-3:]

        x = x.view(-1, movement, feature_num)   # (batch_size, agent_num, 5, 12*7)

        x = self.fc1(x)
        x = self.movement_transformer(x)
        x = x.view(n_envs*batch, timeseries, -1)
        x = self.pooling(x)
        x = self.time_transformer(x)
        x = x.view(n_envs*batch, -1)
        x = self.pooling(x)
        x = self.fc2(x)

        return x.view(n_envs, batch, -1)
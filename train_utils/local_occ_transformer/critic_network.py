'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:56
@Description: Critic Network
@LastEditTime: 2024-04-25 16:27:57
'''
from torch import nn
import torch.nn.functional as F
from .layer import Transformer
from einops import rearrange

# class CriticNetwork(nn.Module):
#     def __init__(self):
#         super(CriticNetwork, self).__init__()
#         self.fc1 = nn.Linear(in_features=60, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=64)
#         self.fc3 = nn.Linear(in_features=64, out_features=1)
#
#     def forward(self, x):
#         x_local = x['local']
#         env_batch_nagents = list(x_local.shape[:-3]) # 包含 n_envs, batchsize 和 n_agents
#         timeseries, movement, feature_num = x_local.shape[-3:]
#
#         x_local = x_local[..., 0].view(-1, timeseries*movement) # 只获得占有率
#         x_local = F.tanh(self.fc1(x_local))
#         x_local = F.tanh(self.fc2(x_local))
#         x_local = self.fc3(x_local)
#
#         return x_local.view(*env_batch_nagents, -1)


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.transformer1 = Transformer(dim=64, depth=2, heads=8, dim_head=128, mlp_dim=128, dropout=0.1)
        self.fc2 = nn.Linear(64, 1)
        self.pooling = nn.AdaptiveAvgPool2d((1, 64))

    def forward(self, x):
        x = x['local']
        if x.ndim == 6:
            eba = list(x.shape[:3])
            x = rearrange(x, 'e b a ... -> (e b a) ...')

        elif x.ndim == 5:
            eba = list(x.shape[:2])
            x = rearrange(x, 'eb a ... -> (eb a) ...')
        else:
            raise 'unexpected ndim'

        time, movement = x.shape[-3:-1]
        x = x[..., 0]  # 只取占有率

        x = self.fc1(x)
        x = self.transformer1(x)
        x = self.pooling(x)
        x = rearrange(x, 'eba 1 d -> eba d')
        x = self.fc2(x)

        return x.view(*eba, -1)

'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:56
@Description: Critic Network
LastEditTime: 2024-09-19 00:13:52
'''
from torch import nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=60, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)
        
    def forward(self, x):
        x_local = x['local']
        env_batch_nagents = list(x_local.shape[:-3]) # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x_local.shape[-3:]

        x_local = x_local[..., 0].view(-1, timeseries*movement) # 只获得占有率
        x_local = F.relu(self.layer_norm1(self.fc1(x_local)))
        x_local = F.relu(self.layer_norm2(self.fc2(x_local)))
        x_local = self.fc3(x_local)

        return x_local.view(*env_batch_nagents, -1)
'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:48
@Description: Actor Network
LastEditTime: 2024-09-19 00:14:48
'''
from torch import nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=60, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=action_size)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)

    def forward(self, x):
        x = x['local'] # 获得局部信息, x['local'], x['vehicle']
        env_batch_nagents = list(x.shape[:-3]) # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x.shape[-3:]

        x = x[..., 0].view(-1, timeseries*movement) # 只获得占有率
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.fc3(x)

        return x.view(*env_batch_nagents, -1)
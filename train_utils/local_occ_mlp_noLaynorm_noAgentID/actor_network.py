'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:48
@Description: Actor Network
LastEditTime: 2024-09-19 02:14:12
'''
from torch import nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=60, out_features=256)  # 5*12=60
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=action_size)

    def forward(self, x):
        x_local = x['local']  # (n_envs, batchsize, n_agents, timeseries, n_movements, n_features)
        env_batch_nagents = list(x_local.shape[:-3])  # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x_local.shape[-3:]

        x_local = x_local[..., 0].view(-1, timeseries * movement)  # 只获得占有率

        # First layer
        out = self.fc1(x_local)
        out = F.relu(out)

        # Second layer
        out = self.fc2(out)
        out = F.relu(out)

        # Third layer
        out = self.fc3(out)
        out = F.relu(out)

        # Output layer
        out = self.fc4(out)

        return out.view(*env_batch_nagents, -1)
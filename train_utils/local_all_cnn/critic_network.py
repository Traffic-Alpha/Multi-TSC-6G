'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:56
@Description: Critic Network
@LastEditTime: 2024-04-25 16:44:14
'''
from torch import nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=(1, 7))
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(in_features=32, out_features=1)

    def forward(self, local_global_x):
        x_local = local_global_x['local']
        # x_global = local_global_x['global']
        env_batch_nagents = list(x_local.shape[:-3]) # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x_local.shape[-3:]
        x_local = x_local.view(-1, timeseries, movement, feature_num) # (batch_size, agent_num, 5, 12, 7)

        x_local = self.conv(x_local) # (batch_size*agent_num, 5, 12, 7) --> (batch_size*agent_num, 32, 12, 1)
        x_local = F.relu(x_local)
        x_local = x_local.squeeze(-1) # (batch_size*agent_num, 32, 12)
        x_local = self.pool(x_local) # (batch_size*agent_num, 32, 1)
        x_local = x_local.squeeze(-1) # (batch_size*agent_num, 32)
        x_local = self.fc(x_local) # (batch_size*agent_num, 2), 2 是动作个数

        return x_local.view(*env_batch_nagents, -1)
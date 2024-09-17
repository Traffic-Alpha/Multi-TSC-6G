'''
@Author: WANG Maonan
@Date: 2024-04-25 16:39:36
@Description: Actor Network 
LastEditTime: 2024-09-17 16:15:26
'''
from torch import nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, action_size):
        super(ActorNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=(1, 7))
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(in_features=32, out_features=action_size)

    def forward(self, x):
        x = x["local"]
        env_batch_nagents = list(x.shape[:-3]) # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x.shape[-3:]

        x = x.view(-1, timeseries, movement, feature_num) # (batch_size*agent_num, 5, 12, 6)

        x = self.conv(x) # (batch_size*agent_num, 5, 12, 7) --> (batch_size*agent_num, 32, 12, 1), 分析 x[2,0].round(decimals=2)
        x = F.tanh(x)
        x = x.squeeze(-1) # (batch_size*agent_num, 32, 12)
        x = self.pool(x) # (batch_size*agent_num, 32, 1)
        x = x.squeeze(-1) # (batch_size*agent_num, 32)
        x = F.tanh(x)
        x = self.fc(x) # (batch_size*agent_num, 3), 3 是动作个数
        return x.view(*env_batch_nagents, -1)
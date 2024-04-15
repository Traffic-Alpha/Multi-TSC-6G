'''
@Author: WANG Maonan
@Date: 2024-04-15 16:09:29
@Description: 创建 Critic Network
1. 中心化的 critic, 也就是使用全局的信息, 最后输出是 1
2. 共享权重, 所有的 agent 的 critic 使用一样的权重
@LastEditTime: 2024-04-15 21:32:40
'''
import torch
from torch import nn
import torch.nn.functional as F

from torchrl.modules import (
    MultiAgentMLP,
    ValueOperator,
)

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
    
class critic_module():
    def __init__(self, device) -> None:
        self.critic_net = CriticNetwork().to(device)

    def make_critic_module(self):
        value_module = ValueOperator(
            module=self.critic_net,
            in_keys=[("agents", "observation")],
        )
        return value_module
    
    def save_model(self, model_path):
        torch.save(self.critic_net.state_dict(), model_path)
    
    def load_model(self, model_path):
        model_dicts = torch.load(model_path)
        self.critic_net.load_state_dict(model_dicts)
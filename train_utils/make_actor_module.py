'''
@Author: WANG Maonan
@Date: 2023-10-30 23:15:18
@Description: 创建 Actor Module
1. 不是中心化, 即每个 agent 根据自己的 obs 进行决策
2. 模型的权重是共享的, 因为 agent 是相同的类型, 所以只有一个 actor 的权重即可
@LastEditTime: 2024-04-15 23:46:19
'''
import torch
from torch import nn
import torch.nn.functional as F

from tensordict.nn import TensorDictModule

from torchrl.data import OneHotDiscreteTensorSpec
from torchrl.modules import (
    OneHotCategorical,
    ProbabilisticActor,
)

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=(1, 7))
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        env_batch_nagents = list(x.shape[:-3]) # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x.shape[-3:]

        x = x.view(-1, timeseries, movement, feature_num) # (batch_size, agent_num, 5, 12, 7)

        x = self.conv(x) # (batch_size*agent_num, 5, 12, 7) --> (batch_size*agent_num, 32, 12, 1)
        x = F.relu(x)
        x = x.squeeze(-1) # (batch_size*agent_num, 32, 12)
        x = self.pool(x) # (batch_size*agent_num, 32, 1)
        x = x.squeeze(-1) # (batch_size*agent_num, 32)
        x = self.fc(x) # (batch_size*agent_num, 2), 2 是动作个数

        return x.view(*env_batch_nagents, -1)


class policy_module():
    def __init__(self, n_agents, device) -> None:
        self.n_agents = n_agents
        self.actor_net = ActorNetwork().to(device)

    def make_policy_module(self):
        policy_module = TensorDictModule(
            self.actor_net,
            in_keys=[("agents", "observation", "local")], # 这里只使用 local 的结果
            out_keys=[("agents", "logits")],
        )
        unbatched_action_spec = OneHotDiscreteTensorSpec(
            n=2, 
            shape=torch.Size([self.n_agents, 2]), 
            dtype=torch.int64
        ) # 2 是指动作空间是 2
        policy = ProbabilisticActor(
            module=policy_module,
            spec=unbatched_action_spec, # CompositeSpec({("agents", "action"): env.action_spec})
            in_keys=[("agents", "logits")],
            out_keys=[("agents", "action")], # env.action_key
            distribution_class=OneHotCategorical,
            return_log_prob=True,
        )
        return policy
    
    def save_model(self, model_path):
        torch.save(self.actor_net.state_dict(), model_path)
    
    def load_model(self, model_path):
        model_dicts = torch.load(model_path)
        self.actor_net.load_state_dict(model_dicts)
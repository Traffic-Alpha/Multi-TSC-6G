'''
@Author: WANG Maonan
@Date: 2024-04-15 16:09:29
@Description: 创建 Critic Network
1. 中心化的 critic, 也就是使用全局的信息, 最后输出是 1
2. 共享权重, 所有的 agent 的 critic 使用一样的权重
@LastEditTime: 2024-04-25 16:48:54
'''
import torch
import importlib

from torchrl.modules import (
    ValueOperator,
)

def load_critic_model(model_name):
    module = importlib.import_module(f'train_utils.{model_name}')
    CriticNetwork = getattr(module, 'CriticNetwork')
    return CriticNetwork
    
class critic_module():
    def __init__(self, model_name, device) -> None:
        CriticNetwork = load_critic_model(model_name)
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
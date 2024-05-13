'''
@Author: WANG Maonan
@Date: 2023-10-30 23:15:18
@Description: 创建 Actor Module
1. 不是中心化, 即每个 agent 根据自己的 obs 进行决策
2. 模型的权重是共享的, 因为 agent 是相同的类型, 所以只有一个 actor 的权重即可
@LastEditTime: 2024-04-25 17:16:49
'''
import torch
import importlib
from loguru import logger
from tensordict.nn import TensorDictModule

from torchrl.modules import (
    OneHotCategorical,
    ProbabilisticActor,
)

def load_actor_model(model_name):
    module = importlib.import_module(f'train_utils.{model_name}')
    ActorNetwork = getattr(module, 'ActorNetwork')
    return ActorNetwork


class policy_module():
    def __init__(self, model_name, action_spec, device) -> None:
        ActorNetwork = load_actor_model(model_name)
        self.action_spec = action_spec
        self.actor_net = ActorNetwork(action_size=action_spec.shape[-1]).to(device)
        logger.info(f'RL: Actor Model:\n {self.actor_net}')

    def make_policy_module(self):
        policy_module = TensorDictModule(
            self.actor_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )
        policy = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys={
                "logits":("agents", "logits"), 
                # "mask":("agents", "action_mask")
            }, # 这里需要传入一个 mask, 根据 agent mask 自己进行计算
            out_keys=[("agents", "action")], # env.action_key
            distribution_class=OneHotCategorical,
            return_log_prob=True,
        )
        return policy
    
    def save_model(self, model_path):
        torch.save(self.actor_net.state_dict(), model_path)
    
    def load_model(self, model_path, device):
        if device is 'cpu':
            model_dicts = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            model_dicts = torch.load(model_path)
        self.actor_net.load_state_dict(model_dicts)
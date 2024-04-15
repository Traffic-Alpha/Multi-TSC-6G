'''
@Author: WANG Maonan
@Date: 2024-04-15 23:41:58
@Description: 加载 MAPPO 模型进行测试
@LastEditTime: 2024-04-15 23:45:47
'''
import torch

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from train_utils.make_actor_module import policy_module
from env_utils.make_multi_tsc_env import make_multi_envs

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sumo_cfg = path_convert("./sumo_nets/3_ints/env/three_junctions.sumocfg")
    net_file = path_convert("./sumo_nets/3_ints/env/three_junctions.net.xml")
    log_path = path_convert('./log/eval')
    n_agents = 3
    
    # 1. Create Env
    tsc_env = make_multi_envs(
        tls_ids=['J1', 'J2', 'J3'],
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=1500,
        use_gui=True,
        log_file=log_path,
        device=device
    )

    # 2. Load Model Dict
    policy_gen = policy_module(n_agents, device)
    policy_gen.load_model(path_convert('mappo_models/actor.pkl'))
    policy = policy_gen.make_policy_module()

    # 3. Simulation with environment using the policy
    rollouts = tsc_env.rollout(
        policy=policy,
        auto_reset=True,
        auto_cast_to_device=True,
        break_when_any_done=False,
        max_steps=1_000
    )
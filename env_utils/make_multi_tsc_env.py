'''
@Author: WANG Maonan
@Date: 2024-04-15 03:58:19
@Description: 创建多智能体的环境
LastEditTime: 2024-09-17 16:27:00
'''
from typing import List, Dict
from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper
from env_utils.pz_env import TSCEnvironmentPZ

from torchrl.envs import (
    ParallelEnv, 
    TransformedEnv,
    RewardSum,
    VecNorm
)
from env_utils.torchrl_pz_wrapper import PettingZooWrapper # 对原始的 torchrl 的 wrapper 进行了修改


def make_multi_envs(
        tls_ids:List[str], 
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
        road_ids:List[str],
        action_space:Dict[str, int],
        cell_length:int,
        log_file:str, device:str='cpu',
        **output_files
    ):
    tsc_env = TSCEnvironment(
        **output_files,
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=num_seconds,
        tls_ids=tls_ids,
        tls_action_type='choose_next_phase_syn',
        use_gui=use_gui,
    )
    tsc_env = GlobalLocalInfoWrapper(tsc_env, filepath=log_file, road_ids=road_ids, cell_length=cell_length)
    tsc_env = TSCEnvironmentPZ(tsc_env, action_space)
    tsc_env = PettingZooWrapper(
        tsc_env, 
        group_map={'agents':tls_ids}, # agent 可以分类, 例如不同动作空间大小
        categorical_actions=False,
        use_mask=False, # 智能体数量动态变化, 手动将 obs 和 reward 设置为 0
        device=device,
        done_on_any=False # 所有都结束才结束
    )
    tsc_env = TransformedEnv(tsc_env)
    tsc_env.append_transform(RewardSum(in_keys=[tsc_env.reward_key]))
    tsc_env.append_transform(VecNorm(in_keys=[tsc_env.reward_key]))

    return tsc_env

def make_parallel_env(
        num_envs:int,
        tls_ids:List[str], 
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
        road_ids:List[str],
        action_space:Dict[str, int],
        cell_length:int,
        log_file:str,
        device:str='cpu'
    ):
    env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=make_multi_envs,
        create_env_kwargs=[{
            "tls_ids": tls_ids,
            "sumo_cfg": sumo_cfg,
            "num_seconds": num_seconds,
            "net_file": net_file,
            "action_space": action_space,
            "road_ids": road_ids,
            "cell_length": cell_length,
            "use_gui" : use_gui,
            "log_file": log_file+f'/{i}',
            "device": device,
        }
        for i in range(num_envs)]
    )

    return env
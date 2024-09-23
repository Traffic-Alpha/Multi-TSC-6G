'''
Author: Maonan Wang
Date: 2024-09-23 13:42:56
LastEditTime: 2024-09-23 15:48:54
LastEditors: Maonan Wang
Description: 测试 rule-based policy 的效果
FilePath: /Multi-TSC-6G/eval_rule_based.py
'''
import json
from typing import List
from functools import partial

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper
from rule_based_policy import (
    ft_policy,
    webster_policy,
    actuated_policy # 感应控制可视化效果很好
)
# 创建不同的 ft_policy 实例
ft_policy_1 = partial(ft_policy, phase_duration=1)
ft_policy_2 = partial(ft_policy, phase_duration=2)
ft_policy_3 = partial(ft_policy, phase_duration=3)

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), file_log_level="INFO", terminal_log_level="INFO")

def load_environment_config(env_config_path):
    env_config_path = path_convert(f'./configs/env_configs/{env_config_path}')
    with open(env_config_path, 'r') as file:
        config = json.load(file)
    return config

def make_multi_envs(
        tls_ids:List[str], 
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
        road_ids: List[str],
        log_file:str, cell_length:int=20
    ):
    tsc_env = TSCEnvironment(
        sumo_cfg=sumo_cfg,
        net_file=net_file, # 用于加载路网的信息
        num_seconds=num_seconds,
        tls_ids=tls_ids,
        tls_action_type='choose_next_phase_syn',
        use_gui=use_gui,
        trip_info=path_convert('./trip_info.xml'),
        statistic_output=path_convert('./statistic_output.xml'),
        summary=path_convert('./summary.xml')
    )
    tsc_env = GlobalLocalInfoWrapper(tsc_env, filepath=log_file, road_ids=road_ids, cell_length=cell_length)

    return tsc_env

if __name__ == '__main__':
    # 指定测试的策略
    eval_policy = webster_policy
    
    # 读取实验配置文件
    env_config = load_environment_config("southKorea_Songdo.json")
    
    sumo_cfg = path_convert(env_config['sumocfg'])
    net_file = path_convert(env_config['sumonet'])
    num_seconds = env_config['simulation_time']
    road_ids = env_config['road_ids']
    log_path = path_convert('./log/')
    env = make_multi_envs(
        tls_ids=['J1', 'J2', 'J3'], # 控制 3 个路口, 都是 2 相位
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=num_seconds,
        road_ids=road_ids,
        log_file=log_path,
        use_gui=True,
        cell_length=50
    )

    done = False
    eposide_reward = 0 # 累计奖励
    states = env.reset()
    (processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask) = states

    # 获得 junction phase 的信息
    junction_phase_group = {} # phase 包含哪些 movement
    junction_movement_ids = env.tls_movement_id.copy() # phase 中体征包含的 movement 的顺序
    for tls_id, tls_info in env.env.tsc_env.scene_objects['tls'].traffic_lights.items():
        junction_phase_group[tls_id] = tls_info.phase2movements.copy()
        
    while not done:
        action = eval_policy(
            traffic_state=processed_local_obs, 
            junction_phase_group=junction_phase_group, 
            junction_movement_ids=junction_movement_ids
        ) # rule-based policy aoutput action
        states, rewards, truncateds, dones, infos = env.step(action)
        (processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask) = states
        eposide_reward += sum(rewards.values()) # 计算累计奖励
        done = all(dones.values()) or (len(dones) == 0)
    env.close()

    print(f"累计奖励为, {eposide_reward}.")
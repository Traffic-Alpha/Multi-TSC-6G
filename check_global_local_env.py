'''
@Author: WANG Maonan
@Date: 2024-04-09 22:32:16
@Description: 检查 Global Local Env, 环境的特征是否提取正确
LastEditTime: 2024-09-17 15:48:12
'''
import json
import numpy as np
from typing import List

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper

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
        use_gui=use_gui
    )
    tsc_env = GlobalLocalInfoWrapper(tsc_env, filepath=log_file, road_ids=road_ids, cell_length=cell_length)

    return tsc_env

if __name__ == '__main__':
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
    state = env.reset()
    while not done:
        action = {
            "J1": np.random.randint(4),
            "J2": np.random.randint(4),
            "J3": np.random.randint(4),
        }
        states, rewards, truncateds, dones, infos = env.step(action)
        (processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask) = states
        eposide_reward += sum(rewards.values()) # 计算累计奖励
        done = all(dones.values()) or (len(dones) == 0)
    env.close()

    print(f"累计奖励为, {eposide_reward}.")
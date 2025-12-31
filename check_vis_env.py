'''
@Author: WANG Maonan
@Date: 2024-04-09 22:32:16
@Description: 检查 Global Local Env 和 Vis Env
=> Global Local Env: 环境的特征是否提取正确
=> Vis Env: 是否可以正确进行可视化
LastEditTime: 2025-10-16 17:42:56
'''
import json
import numpy as np
from typing import List

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper
from env_utils.vis_wrapper import VisWrapper

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

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
        tls_action_type='choose_next_phase',
        use_gui=use_gui
    )
    tsc_env = GlobalLocalInfoWrapper(tsc_env, filepath=log_file, road_ids=road_ids, cell_length=20)
    tsc_env = VisWrapper(tsc_env) # 加入绘制全局特征的功能

    return tsc_env

if __name__ == '__main__':
    # 读取配置文件
    env_config = load_environment_config("southKorea_Songdo.json")

    sumo_cfg = path_convert(env_config['sumocfg'])
    net_file = path_convert(env_config['sumonet'])
    num_seconds = env_config['simulation_time']
    road_ids = env_config['road_ids']
    log_path = path_convert('./log/')
    
    env = make_multi_envs(
        tls_ids=['J1', 'J2', 'J3'],
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=num_seconds,
        use_gui=True,
        road_ids=road_ids,
        log_file=log_path,
        cell_length=50,
    )

    done = False
    state = env.reset()
    while not done:
        action = {
            "J1": np.random.randint(4),
            "J2": np.random.randint(4),
            "J3": np.random.randint(4),
        }
        env.step(action)
        # env.plot_map(timestamp=120, attributes=['total_vehicles', 'average_waiting_time', 'average_speed']) # 需要加入 vis_wrapper 之后才可以有的功能
        # env.plot_edge_attribute(edge_id='-1105574288#1', attribute='vehicles')
        # env.plot_edge_attribute(edge_id='-1105574288#1', attribute='average_waiting_time')
        # env.plot_edge_attribute(edge_id='-23755720#5', attribute='average_waiting_time')
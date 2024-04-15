'''
@Author: WANG Maonan
@Date: 2024-04-09 22:32:16
@Description: 检查 Global Local Env 和 Vis Env
=> Global Local Env: 环境的特征是否提取正确
=> Vis Env: 是否可以正确进行可视化
@LastEditTime: 2024-04-14 18:13:58
'''
import numpy as np
from typing import List

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper
from env_utils.vis_wrapper import VisWrapper

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def make_multi_envs(
        tls_ids:List[str], 
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
    ):
    tsc_env = TSCEnvironment(
        sumo_cfg=sumo_cfg,
        net_file=net_file, # 用于加载路网的信息
        num_seconds=num_seconds,
        tls_ids=tls_ids,
        tls_action_type='choose_next_phase',
        use_gui=use_gui
    )
    tsc_env = GlobalLocalInfoWrapper(tsc_env, cell_length=20)
    tsc_env = VisWrapper(tsc_env) # 加入绘制全局特征的功能

    return tsc_env

if __name__ == '__main__':
    sumo_cfg = path_convert("./sumo_nets/osm_berlin/env/berlin.sumocfg")
    net_file = path_convert("./sumo_nets/osm_berlin/env/berlin.net.xml")
    log_path = path_convert('./log/osm_berlin')
    env = make_multi_envs(
        tls_ids=[
            "25663405", "25663407", 
            "25663423", "25663436", 
            "25663429", "25663426"
        ], # 控制 6 个路口
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=1000,
        use_gui=True,
    )

    done = False
    state = env.reset()
    while not done:
        action = {
            "25663405": np.random.randint(4), "25663407": np.random.randint(4), 
            "25663423": np.random.randint(4), "25663436": np.random.randint(3), 
            "25663429": np.random.randint(4), "25663426": np.random.randint(4)
        }
        env.step(action)
        # env.plot_map(timestamp=120, attributes=['total_vehicles', 'average_waiting_time', 'average_speed']) # 需要加入 vis_wrapper 之后才可以有的功能
        # env.plot_edge_attribute(edge_id='-1105574288#1', attribute='vehicles')
        # env.plot_edge_attribute(edge_id='-1105574288#1', attribute='average_waiting_time')
        # env.plot_edge_attribute(edge_id='-23755720#5', attribute='average_waiting_time')
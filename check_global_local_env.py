'''
@Author: WANG Maonan
@Date: 2024-04-09 22:32:16
@Description: 检查 Global Local Env, 环境的特征是否提取正确
@LastEditTime: 2024-04-24 22:26:08
'''
import numpy as np
from typing import List

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def make_multi_envs(
        tls_ids:List[str], 
        sumo_cfg:str, net_file:str,
        num_seconds:int, use_gui:bool,
    ):
    log_file = path_convert('./log/3_ints_wrapper')
    tsc_env = TSCEnvironment(
        sumo_cfg=sumo_cfg,
        net_file=net_file, # 用于加载路网的信息
        num_seconds=num_seconds,
        tls_ids=tls_ids,
        tls_action_type='choose_next_phase_syn',
        use_gui=use_gui
    )
    tsc_env = GlobalLocalInfoWrapper(tsc_env, filepath=log_file, cell_length=20)

    return tsc_env

if __name__ == '__main__':
    sumo_cfg = path_convert("./sumo_nets/3_ints/env/three_junctions.sumocfg")
    net_file = path_convert("./sumo_nets/3_ints/env/three_junctions.net.xml")
    log_path = path_convert('./log/3_ints')
    env = make_multi_envs(
        tls_ids=['J1', 'J2', 'J3'], # 控制 3 个路口, 都是 2 相位
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=2000,
        use_gui=False,
    )

    done = False
    state = env.reset()
    while not done:
        action = {
            "J1": np.random.randint(2),
            "J2": np.random.randint(2),
            "J3": np.random.randint(2),
        }
        states, rewards, truncateds, dones, infos = env.step(action)
        (processed_local_obs, processed_global_obs) = states
        done = all(dones.values()) or (len(dones) == 0)
    env.close()
'''
@Author: WANG Maonan
@Date: 2024-04-14 17:56:50
@Description: 检查 petting zoo 的环境
@LastEditTime: 2024-04-15 03:55:50
'''
import numpy as np
from typing import List
from pettingzoo.test import parallel_api_test

from env_utils.tsc_env import TSCEnvironment
from env_utils.global_local_wrapper import GlobalLocalInfoWrapper
from env_utils.pz_env import TSCEnvironmentPZ

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def make_pz_envs(
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
    tsc_env = TSCEnvironmentPZ(tsc_env)

    return tsc_env

if __name__ == '__main__':
    sumo_cfg = path_convert("./sumo_nets/3_ints/env/three_junctions.sumocfg")
    net_file = path_convert("./sumo_nets/3_ints/env/three_junctions.net.xml")
    log_path = path_convert('./log/3_ints')
    env = make_pz_envs(
        tls_ids=['J1', 'J2', 'J3'], # 控制 3 个路口, 都是 2 相位
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=2000,
        use_gui=True,
    )

    # parallel_api_test(env, num_cycles=1_000_000) # 这种 agent 变化会导致无法通过 test, 但是可以在 torchrl 里面使用
    for _ in range(3): # 完整运行三次仿真, 查看是否有出错
        state, info = env.reset()
        dones = False
        while not dones:
            random_action = {_tls_id:np.random.randint(2) for _tls_id in ["J1", "J2", "J3"]}
            observations, rewards, terminations, truncations, infos = env.step(random_action)
            done = all(terminations.values())
    env.close()
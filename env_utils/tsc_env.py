'''
@Author: WANG Maonan
@Date: 2023-10-29 23:26:57
@Description: 6G 环境下多路口信号灯控制环境
@LastEditTime: 2024-05-07 00:45:41
'''
import gymnasium as gym
from typing import List, Dict
from tshub.tshub_env.tshub_env import TshubEnvironment

class TSCEnvironment(gym.Env):
    def __init__(
            self, sumo_cfg:str, 
            net_file:str,
            num_seconds:int, 
            tls_ids:List[str], 
            tls_action_type:str, 
            use_gui:bool=False,
            **output_files
        ) -> None:
        super().__init__()

        self.tls_ids = tls_ids # 需要控制的路口 ID
        self.tsc_env = TshubEnvironment(
            sumo_cfg=sumo_cfg,
            net_file=net_file,
            is_aircraft_builder_initialized=False, 
            is_vehicle_builder_initialized=True, # reward & glocal info
            is_traffic_light_builder_initialized=True, # get local info
            is_map_builder_initialized=True, # 需要初始化地图, 获得 lane 的信息 (for global info)
            is_person_builder_initialized=False,
            tls_ids=tls_ids,
            num_seconds=num_seconds,
            tls_action_type=tls_action_type,
            use_gui=use_gui,
            is_libsumo=(not use_gui), # 如果不开界面, 就是用 libsumo
            **output_files
        )

    def reset(self):
        state_infos = self.tsc_env.reset()
        return state_infos
        
    def step(self, action:Dict[str, Dict[str, int]]):
        action = {'tls': action} # 这里只控制 tls 即可
        states, rewards, infos, dones = self.tsc_env.step(action)
        truncated = dones

        return states, rewards, truncated, dones, infos
    
    def close(self) -> None:
        self.tsc_env._close_simulation()
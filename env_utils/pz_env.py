'''
@Author: WANG Maonan
@Date: 2024-04-14 17:15:14
@Description: Petting Zoo Wrapper
=> 关于 petting zoo 环境创建, 参考 
    1. https://pettingzoo.farama.org/content/environment_creation/
    2. https://pettingzoo.farama.org/tutorials/custom_environment/3-action-masking/ (添加 action mask)
=> 由于不是每一个时刻所有 TSC 都可以做动作, 这里我们就只返回可以做动作的 TSC 的信息, 也就是 agent 的数量是一直在改变的
LastEditTime: 2024-09-17 17:15:32
'''
import functools
import numpy as np
import gymnasium as gym

from typing import Dict
from pettingzoo import ParallelEnv

class TSCEnvironmentPZ(ParallelEnv):
    metadata = {
        "name": "multi_agent_tsc_env",
    }
    
    def __init__(self, env, action_space):
        super().__init__()
        self.env = env
        self.render_mode = None
        
        # agent id == tls id (agent id 就是所有信号灯的 id)
        self.possible_agents = self.env.tls_ids
        self.agents = self.possible_agents.copy()

        # spaces
        self.action_spaces = {
            _tls_id:gym.spaces.Discrete(action_space[_tls_id]) # 每一个信号灯的相位个数
            for _tls_id in self.env.tls_ids
        }
        self.observation_spaces = {
            _tls_id:gym.spaces.Dict({
                "agent_id": gym.spaces.Box(low=0, high=len(self.agents)), # 表明 agent id, 为了区分不同的 agent
                "local": gym.spaces.Box(
                    low=np.zeros((5,12,7)),
                    high=np.ones((5,12,7)),
                    shape=(5,12,7,)
                ),
                # "global": gym.spaces.Box(
                #     low=np.zeros((294,5,56,3)), # 3ints-(20,5,11,3)
                #     high=np.ones((294,5,56,3)),
                #     shape=(294,5,56,3,)
                # ), # 20 个 edge, 每个 edge 包含 5s 的数据, 每个 edge 有 11 个 cell, 每个 cell 有 3 个信息
                # "global_mask": gym.spaces.Box(
                #     low=np.zeros((294,56)), # 3int-(20,11)
                #     high=np.ones((294,56)),
                #     shape=(294,56)
                # ),
                # "vehicle": gym.spaces.Box(
                #     low=np.zeros((5,100,299)), # TODO, 这里车辆的 road id 也是需要修改的
                #     high=100*np.ones((5,100,299)),
                #     shape=(5,100,299)
                # ),
                # "vehicle_mask": gym.spaces.Box(
                #     low=np.zeros((5,100)),
                #     high=np.ones((5,100)),
                #     shape=(5,100)
                # ),                
            })
            for _tls_id in self.env.tls_ids
        }

    def reset(self, seed=None, options=None):
        """Reset the environment
        """
        processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask = self.env.reset()
        agent_mask = {
            _tls_id:{
                'can_perform_action': True, 
            }
            for _tls_id in self.possible_agents
        }
        self.agents = self.possible_agents[:] # 可以做动作的 agent

        # 处理 observation
        observations = {
            _tls_id: {
                'agent_id': _tls_index,
                'local': processed_local_obs[_tls_id],
                # 'global': processed_global_obs,
                # 'global_mask': edge_cell_mask,
                # 'vehicle': processed_veh_obs[_tls_id],
                # 'vehicle_mask': processed_veh_mask[_tls_id]
            }
            for _tls_index, _tls_id in enumerate(self.agents)
        }

        return observations, agent_mask
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Return the observation space for the agent.
        """
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Return the action space for the agent.
        """
        return self.action_spaces[agent]
    
    def close(self):
        """Close the environment and stop the SUMO simulation.
        """
        self.env.close()

    def step(self, actions:Dict[str, int]):
        """Step the environment.
        """
        (processed_local_obs, processed_global_obs, edge_cell_mask, processed_veh_obs, processed_veh_mask), rewards, terminations, truncations, infos = self.env.step(actions)

        # 将不能做动作的 agent 设置为 0
        pz_observations = {}
        pz_rewards = {}
        
        # 处理 observation
        for _tls_index, _tls_id in enumerate(self.possible_agents):
            pz_observations[_tls_id] = {
                'agent_id': _tls_index,
                'local': processed_local_obs[_tls_id],
                # 'global': processed_global_obs,
                # 'global_mask': edge_cell_mask,
                # 'vehicle': processed_veh_obs[_tls_id],
                # 'vehicle_mask': processed_veh_mask[_tls_id]
            }
            pz_rewards[_tls_id] = rewards[_tls_id]

        
        return pz_observations, pz_rewards, terminations, truncations, infos
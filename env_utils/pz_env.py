'''
@Author: WANG Maonan
@Date: 2024-04-14 17:15:14
@Description: Petting Zoo Wrapper
=> 关于 petting zoo 环境创建, 参考 
    1. https://pettingzoo.farama.org/content/environment_creation/
    2. https://pettingzoo.farama.org/tutorials/custom_environment/3-action-masking/ (添加 action mask)
=> 由于不是每一个时刻所有 TSC 都可以做动作, 这里我们就只返回可以做动作的 TSC 的信息, 也就是 agent 的数量是一直在改变的
@LastEditTime: 2024-04-17 03:52:26
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
        
    def __init__(self, env):
        super().__init__()
        self.env = env
        
        # agent id == tls id (agent id 就是所有信号灯的 id)
        self.possible_agents = self.env.tls_ids
        self.agents = self.possible_agents.copy()

        # spaces
        self.action_spaces = {
            _tls_id:gym.spaces.Discrete(3) # 最后一个动作专门为了跳过当前的状态 
            for _tls_id in self.env.tls_ids
        } # 这里都是 2 相位的信号灯
        self.observation_spaces = {
            _tls_id:gym.spaces.Dict({
                "local": gym.spaces.Box(
                    low=np.zeros((5,12,7)),
                    high=np.ones((5,12,7)),
                    shape=(5,12,7,)
                ),
                "global": gym.spaces.Box(
                    low=np.zeros((20,5,11,3)),
                    high=np.ones((20,5,11,3)),
                    shape=(20,5,11,3,)
                ), # 20 个 edge, 每个 edge 包含 5s 的数据, 每个 edge 有 11 个 cell, 每个 cell 有 3 个信息
            })
            for _tls_id in self.env.tls_ids
        } # 这里都是 2 相位的信号灯

    def reset(self, seed=None, options=None):
        """Reset the environment
        """
        processed_local_obs, processed_global_obs = self.env.reset()
        agent_mask = {
            _tls_id:{
                'can_perform_action': True, 
                "action_mask": [True, True, False]
            }
            for _tls_id in self.possible_agents
        }
        self.agents = self.possible_agents[:] # 可以做动作的 agent

        # 处理 observation
        observations = {
            _tls_id: {
                'local': processed_local_obs[_tls_id],
                'global': processed_global_obs,
                }
            for _tls_id in self.agents
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
        (processed_local_obs, processed_global_obs), rewards, terminations, truncations, infos = self.env.step(actions)
        
        # agent 数量发生改变 (但是这里我自己手动把不存在的 agent 的信息设置为 0)
        live_agents = []
        for _tls_id in self.possible_agents:
            if infos[_tls_id]['can_perform_action']:
                live_agents.append(_tls_id) # 存活的 agent
                infos[_tls_id]['action_mask'] = [True, True, False] # Note: 最后一个 action 是留给不能做动作的 agent
            else:
                infos[_tls_id]['action_mask'] = [False, False, True]

        # 将不能做动作的 agent 设置为 0
        pz_observations = {}
        pz_rewards = {}
        
        # 处理 observation
        for _tls_id in self.possible_agents:
            if _tls_id in live_agents:
                pz_observations[_tls_id] = {
                    'local': processed_local_obs[_tls_id],
                    'global': processed_global_obs,
                }
                pz_rewards[_tls_id] = rewards[_tls_id]
            else:
                pz_observations[_tls_id] = {
                    'local': np.zeros((5,12,7)),
                    'global': np.zeros((20,5,11,3)),
                }   
                pz_rewards[_tls_id] = 0             

        
        return pz_observations, pz_rewards, terminations, truncations, infos
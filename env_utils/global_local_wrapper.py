'''
@Author: WANG Maonan
@Date: 2024-04-10 00:21:49
@Description: 根据 state 提取 global info 和 local info, 具体来说分为三类特征:
1. 微观特征: 车辆的属性
2. 中观特征: 路段摄像头的数据
3. 宏观特征: 6G as a sensor
@LastEditTime: 2024-05-07 21:11:32
'''
import time
import numpy as np
import gymnasium as gym
from typing import Dict, Any
from stable_baselines3.common.monitor import ResultsWriter

from ._utils import (
    TimeSeriesData, 
    direction_to_flags, 
    merge_local_data, 
    one_hot_encode,
    calculate_distance
)


class GlobalLocalInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, filepath:str, cell_length:float=20):
        super().__init__(env)
        self.tls_ids = self.env.tls_ids # 多路口的 ids
        self.cell_length = cell_length # 每个 cell 的长度
        self.filepath = filepath
        self.max_vehicle_num = 100 # 记录每个路口的 max_vehicle_num 数量的车

        # 记录时序数据
        self.vehicle_timeseries = TimeSeriesData(N=10) # 记录车辆的数据
        self.vehicle_masks_timeseries = TimeSeriesData(N=10)
        self.local_obs_timeseries = TimeSeriesData(N=10) # 将局部信息全部保存起来
        self.edge_cells_timeseries = TimeSeriesData(N=10) # edge cell 的结果

        # reset init
        self.node_infos = None # 节点的信息
        self.lane_infos = None # 地图车道的原始信息
        self.road_ids = None # 地图所有的 edge id, (1). 用于记录 global feature 的顺序; (2). 用于做 one-hot 向量, 表明车辆的位置
        self.tls_nodes = None # 每个信号灯的 id 和 坐标
        self.vehicle_feature_dim = 0 # 车辆特征的维度
        self.tls_movement_id = {} # 每个路口的 movement 顺序
        self.max_num_cells = -1 # 最大的 cell 个数, 确保 global info 大小一样
        self.edge_cell_mask = [] # 记录 global cell 的时候, 哪些是 padding 的

        # #######
        # Writer
        # #######
        if self.filepath is not None:
            self.t_start = time.time()
            self.results_writer = ResultsWriter(
                    filepath,
                    header={"t_start": self.t_start},
            )
            self.rewards_writer = list()

    def __initialize_edge_cells(self):
        """根据道路信息初始化 edge cell 的信息, 这里每个时刻都需要进行初始化
        """
        edge_cells = {}
        for lane_id, lane_info in self.lane_infos.items():
            edge_id = lane_info['edge_id']
            # num_cells = int(lane_info['length'] // self.cell_length)+1
            assert  self.max_num_cells != -1, '检查代码关于 max_num_cells 的部分, 现在是 -1.'
            num_cells = int(self.max_num_cells) # 统一大小, 这样不同 lane 的长度是一样的
            if edge_id not in edge_cells:
                edge_cells[edge_id] = [
                    {
                        'vehicles': 0, 
                        'total_waiting_time': 0.0, 
                        'total_speed': 0.0, 
                        'total_co2_emission': 0.0,
                    } for _ in range(num_cells)
                ] # 初始化每一个 cell 的信息
        return edge_cells

    # ####################
    # 下面开始处理每一步的特征
    # ####################
    def get_vehicle_obs(self, vehicle_data:Dict[str, Dict[str, Any]]):
        """用于获得每一辆车的信息, 主要包含:
        1. 车辆的速度
        2. 车辆所在的 road, 使用 one-hot
        3. 车辆所在的 lane position
        4. 车辆的 waiting time
        5. 车辆的 accumulated_waiting_time
        6. 车辆记录路口的距离

        这里我们找出与每个路口最接近的 self.max_vehicle_num 辆车

        Args:
            vehicle_data (Dict[str, Dict[str, Any]]): 仿真中车辆的信息, 下面是一个例子:
            {
                '-E6__0__background_1.0': {'id': '-E6__0__background_1.0', 'action_type': 'lane', 'vehicle_type': 'background_1', 'length': 7.0, 'width': 1.8, 'heading': 355.69461230491356, 'position': (293.83316888099887, -59.30507987749093), 'speed': 4.603267983696424, 'road_id': '-E6', 'lane_id': '-E6_1', 'lane_index': 1, 'lane_position': 13.943846113653855, 'edges': ('-E6', 'E2', 'E7'), 'waiting_time': 0.0, ...},
                '-E7__0__background_2.0': {'id': '-E7__0__background_2.0', 'action_type': 'lane', 'vehicle_type': 'background_2', 'length': 7.0, 'width': 1.8, 'heading': 269.75716619082107, 'position': (658.4361538863461, 49.21090214725246), 'speed': 4.603267983696424, 'road_id': '-E7', 'lane_id': '-E7_0', 'lane_index': 0, 'lane_position': 13.943846113653855, 'edges': ('-E7', '-E2', '-E1', '-E0'), 'waiting_time': 0.0, ...}，
                ...
            }
        """
        closest_vehicles = {intersection: [] for intersection in self.tls_ids}
        padding_masks = {intersection: [] for intersection in self.tls_ids}
        
        # Step 2: Calculate distance of each vehicle from each intersection
        for veh_id, veh_data in vehicle_data.items():
            _lane_id = veh_data['lane_id']
            if not _lane_id.startswith(":"):
                for intersection_id, intersection_pos in self.tls_nodes.items():
                    _distance = calculate_distance(veh_data['position'], intersection_pos)
                    _speed = veh_data['speed']
                    _lane_position = veh_data['lane_position']/self.lane_infos[_lane_id]['length']
                    _waiting_time = veh_data['waiting_time']
                    _accumulated_waiting_time = veh_data['accumulated_waiting_time']
                    _edge_id = one_hot_encode(self.road_ids, veh_data['road_id'])
                    closest_vehicles[intersection_id].append([_distance, _speed, _lane_position, _waiting_time, _accumulated_waiting_time] + _edge_id)
                    padding_masks[intersection_id].append(1)  # 1 indicates actual vehicle data

        # Step 3: Sort the vehicles by distance for each intersection and take the closest N
        for intersection_id in self.tls_nodes:
            closest_vehicles[intersection_id].sort(key=lambda x: x[0]) # 按照距离进行排序
            closest_vehicles[intersection_id] = closest_vehicles[intersection_id][:self.max_vehicle_num] # 只取前 max_vehicle_num 个
            padding_masks[intersection_id] = padding_masks[intersection_id][:self.max_vehicle_num]
            
            # Padding if there are less than N vehicles
            num_padding = self.max_vehicle_num - len(closest_vehicles[intersection_id])
            closest_vehicles[intersection_id].extend([[0] * self.vehicle_feature_dim] * num_padding) # self.vehicle_feature_dim 在 reset 的时候计算
            padding_masks[intersection_id].extend([0] * num_padding)  # 0 indicates padding

        return closest_vehicles, padding_masks

    def get_edge_cells(self, vehicle_data:Dict[str, Dict[str, Any]]):
        """计算每一个 edge cell 每一个时刻的信息, 可以用于计算 global info, 或是用于可视化

        Args:
            vehicle_data (Dict[str, Dict[str, Any]]): 仿真中车辆的信息, 具体例子见上面

        Returns:
            _type_: _description_
        """
        edge_cells = self.__initialize_edge_cells() # 初始化 cell 信息

        # 首先统计当前时刻 vehicle 在哪一个 cell, 然后改变 cell 的统计量
        for vehicle_id, vehicle_info in vehicle_data.items():
            edge_id = vehicle_info['road_id']
            if not edge_id.startswith(':'): # 不考虑交叉路口里面
                lane_position = vehicle_info['lane_position']
                cell_index = int(lane_position // self.cell_length) # 计算属于哪一个 cell
                
                cell = edge_cells[edge_id][cell_index]
                cell['vehicles'] += 1
                cell['total_waiting_time'] += vehicle_info['waiting_time']
                cell['total_speed'] += vehicle_info['speed']
                cell['total_co2_emission'] += vehicle_info['co2_emission']

        # 最后输出的时候计算平均值即可
        for edge_id, cells in edge_cells.items():
            for cell in cells:
                if cell['vehicles'] > 0:
                    cell['average_waiting_time'] = cell['total_waiting_time'] / cell['vehicles']
                    cell['average_speed'] = cell['total_speed'] / cell['vehicles']
                    cell['average_co2_emission'] = cell['total_co2_emission'] / cell['vehicles']
                else:
                    cell['average_waiting_time'] = 0.0
                    cell['average_speed'] = 0.0
                    cell['average_co2_emission'] = 0.0

        return edge_cells

    def get_local_tls_state(self, tls_states):
        """获得每个路口每一个时刻的信息, 获得摄像头的数据
        """
        tls_local_obs = {} # 每一个 tls 处理好的特征

        for _tls_id in self.tls_ids: # 依次处理每一个路口
            process_local_obs = []
            for _movement_index, _movement_id in enumerate(self.tls_movement_id[_tls_id]):
                occupancy = tls_states[_tls_id]['last_step_occupancy'][_movement_index]/100
                mean_speed = tls_states[_tls_id]['last_step_mean_speed'][_movement_index] # 获得平均速度
                direction_flags = direction_to_flags(tls_states[_tls_id]['movement_directions'][_movement_id])
                lane_numbers = tls_states[_tls_id]['movement_lane_numbers'][_movement_id]/5 # 车道数 (默认不会超过 5 个车道)
                is_now_phase = int(tls_states[_tls_id]['this_phase'][_movement_index])
                # 将其添加到 obs 中
                process_local_obs.append([occupancy, mean_speed, *direction_flags, lane_numbers, is_now_phase]) # 某个 movement 对应的信息
                
            # 不是四岔路, 进行不全
            for _ in range(12 - len(process_local_obs)):
                process_local_obs.append([0]*len(process_local_obs[0]))
            
            tls_local_obs[_tls_id] = process_local_obs # 存储每个路口处理好的信息

        return tls_local_obs


    # #################
    # 下面开始处理时序特征
    # #################
    def process_global_state(self, K=5):
        """根据 edge cell 的信息来计算 global info, 每个 cell 都是一个向量, 同时包含 cell 的坐标
        """
        _recent_k_data = self.edge_cells_timeseries.get_recent_k_data(K)
        result = {id_key: [] for _, id_data in _recent_k_data for id_key in id_data}
        
        # Iterate over the input data
        for time, id_data in _recent_k_data:
            for edge_id, cell_data in id_data.items(): # 每个 edge 的数据
                edge_info = []
                for _cell_info in cell_data: # 某个时刻, 某个 edge 对应的 cell 的数据
                    _cell_vehicle = _cell_info['vehicles']/2
                    _cell_avg_waiting_time = _cell_info['average_waiting_time']
                    _cell_speed = _cell_info['average_speed']
                    edge_info.append([_cell_vehicle, _cell_avg_waiting_time, _cell_speed])
                result[edge_id].append(edge_info)
        
        # stack
        final_result = []
        for id_key in self.road_ids:
            final_result.append(result[id_key])
        
        return np.stack(final_result)

    def process_local_state(self, K=5):
        """计算局部的信息, 需要可以处理 reset 的情况, 也就是可以处理时间序列不全的时候
        """
        _recent_k_data = self.local_obs_timeseries.get_recent_k_data(K)
        return merge_local_data(_recent_k_data)
    
    def process_veh_state(self, K=5):
        """获得最后 K 个时刻车辆的信息

        Args:
            K (int, optional): 去 K 个时间片, 这里车辆的信息只使用最后一个时刻的信息. Defaults to 5.
        """
        _recent_k_data_veh = self.vehicle_timeseries.get_recent_k_data(K)
        _recent_k_data_veh_padding = self.vehicle_masks_timeseries.get_recent_k_data(K)
        return merge_local_data(_recent_k_data_veh), merge_local_data(_recent_k_data_veh_padding)

    def process_reward(self, vehicle_state):
        """
        Calculate the average waiting time for vehicles at all intersections.
        这里是按整个路网计算一个统一的奖励, 而不是每一个路口计算名一个奖励

        :param vehicle_state: The state of vehicles in the environment.
        :return: The negative average waiting time as the reward.
        """
        waiting_times = [veh['waiting_time'] for veh in vehicle_state.values()]
        
        return -np.mean(waiting_times) if waiting_times else 0

    # #############
    # reset & step
    # #############
    def reset(self, seed=1):
        """reset env
        """
        state = self.env.reset()
        self.node_infos = state['node'] # 地图节点的信息
        self.lane_infos = state['lane'] # 地图车道信息
        self.road_ids = sorted(set([_lane['edge_id'] for _,_lane in self.lane_infos.items()])) # 获得所有的 edge id, 用于对车辆所在位置进行 one-hot
        self.tls_nodes = {_node_id:self.node_infos[_node_id]['node_coord'] for _node_id in self.tls_ids} # 找到所有信号灯对应的坐标
        self.vehicle_feature_dim = 5 + len(self.road_ids) # 车辆的特征的维度

        # 更新全局最大的 max_num_cells 的个数, 同时记录每一个 edge 的 cell 个数
        _edge_cell_num = {}
        self.edge_cell_mask = []
        for _, lane_info in self.lane_infos.items():
            _edge_id = lane_info['edge_id']
            _num_cell = lane_info['length'] // self.cell_length + 1
            _edge_cell_num[_edge_id] = _num_cell # 更新 edge_id 对应的 cell 数量
            if _num_cell > self.max_num_cells:
                self.max_num_cells = _num_cell
        # 更新 global mask
        for _road_id in self.road_ids:
            _num_cell = _edge_cell_num[_road_id]
            _road_mask = [1]*int(_num_cell) + [0]*int(self.max_num_cells-_num_cell)
            self.edge_cell_mask.append(_road_mask)

        # Update Local Info
        tls_data = state['tls'] # 获得路口的信息
        for _tls_id, _tls_data in tls_data.items():
            self.tls_movement_id[_tls_id] = _tls_data['movement_ids'] # 获得每个 tls 的 movement id
        local_obs = self.get_local_tls_state(tls_data)
        self.local_obs_timeseries.add_data_point(timestamp=0, data=local_obs) # 记录局部信息

        # Update Global Info
        vehicle_data = state['vehicle'] # 获得车辆的信息
        global_obs = self.get_edge_cells(vehicle_data) # 得到每一个 cell 的信息
        self.edge_cells_timeseries.add_data_point(timestamp=0, data=global_obs) # 记录全局信息
        
        # Vehicle Local Info
        vehicle_obs, padding_masks = self.get_vehicle_obs(vehicle_data)
        self.vehicle_timeseries.add_data_point(timestamp=0, data=vehicle_obs)
        self.vehicle_masks_timeseries.add_data_point(timestamp=0, data=padding_masks)

        # TODO, 这里需要根据不同的路网手动调整 (这里可以使用参数传入, 参数在配置文件里面, 每一个环境的参数应该是固定的)
        # TODO, reset 状态是随机的, 随机会比全 0 效果好一些
        processed_local_obs = {_tls_id:np.random.randn(5,12,7) for _tls_id in self.tls_ids} # 5 是时间序列, 12 movement 数量, 6 是每个 movement 的特征
        processed_global_obs = np.random.randn(len(global_obs),5,int(self.max_num_cells),3) # len(global_obs): edge 的数量, 5 是时间序列, self.max_num_cells 是 cell 数量, 3 是每个 edge 的特征
        processed_veh_obs = {_tls_id:np.random.randn(5,100,25) for _tls_id in self.tls_ids}
        processed_veh_mask = {_tls_id:np.zeros((5,100)) for _tls_id in self.tls_ids}

        return (processed_local_obs, processed_global_obs, np.array(self.edge_cell_mask), processed_veh_obs, processed_veh_mask)
    

    def step(self, action: Dict[str, int]):
        """这里的数据处理流程为:
        => 获取每个时刻的信息
        1. self.get_local_tls_state, 获取每一个时刻信号灯的信息
        2. self.get_edge_cells, 获取每一个时刻每一个 cell 的信息
        3. self.get_vehicle_obs, 获得每一个时刻每一辆车的信息
        4. 这里会使用 self.xxx_timeseries.add_data_point, 将每个时刻的数据保存起来
        => 将每一个时刻拼接为时间序列
        1. self.process_local_state, 拼接局部信息
        2. self.process_global_state, 拼接全局信息
        3. self.process_veh_state, 拼接车辆的信息
        """
        can_perform_actions = {_tls_id:False for _tls_id in self.tls_ids}

        # 与环境交互
        while not any(can_perform_actions.values()): # 有一个信号灯要做动作, 就需要进行下发
            states, rewards, truncated, done, infos = super().step(action)
            # 1. 获得 states 中车辆和信号灯的信息
            vehicle_data = states['vehicle'] # 获得车辆的信息
            tls_data = states['tls'] # 获得路口的信息
            sim_step = int(self.env.tsc_env.sim_step) # 仿真时间

            # 2. 更新 can_perform_action
            can_perform_actions = {
                _tls_id: tls_data[_tls_id]['can_perform_action']
                for _tls_id in self.tls_ids
            } # 只要有一个 traffic signal 可以做动作, 就需要进行下发
            
            # 3. 根据原始信息提取 (1) veh obs. (2) local obs. (3) global obs
            veh_obs, padding_masks = self.get_vehicle_obs(vehicle_data) # Get Vehicle Info
            self.vehicle_timeseries.add_data_point(sim_step, veh_obs)
            self.vehicle_masks_timeseries.add_data_point(sim_step, data=padding_masks)
            local_obs = self.get_local_tls_state(tls_data) # Get Local Info
            self.local_obs_timeseries.add_data_point(sim_step, local_obs) 
            global_obs = self.get_edge_cells(vehicle_data) # Get Edge Cell
            self.edge_cells_timeseries.add_data_point(sim_step, global_obs)

        # 开始处理 state, 1. local; 2. local+global; 3. local+global+vehicle
        processed_local_obs = self.process_local_state()
        processed_global_obs = self.process_global_state()
        processed_veh_obs, processed_veh_mask = self.process_veh_state()
        
        # 处理 reward
        reward = self.process_reward(vehicle_data) # 计算路网的整体等待时间
        rewards = {_tls_id:reward for _tls_id in self.tls_ids}

        # 处理 dones & truncateds
        dones = {_tls_id:done for _tls_id in self.tls_ids}
        truncateds = {_tls_id:truncated for _tls_id in self.tls_ids}

        # 处理 info
        infos = {_tls_id:{} for _tls_id in self.tls_ids}
        for _tls_id, _can_perform in can_perform_actions.items():
            infos[_tls_id]['can_perform_action'] = _can_perform

        # Writer
        if self.filepath is not None:
            self.rewards_writer.append(float(sum(rewards.values())))
            if all(dones.values()):
                ep_rew = sum(self.rewards_writer)
                ep_len = len(self.rewards_writer)
                ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                self.results_writer.write_row(ep_info)
                self.rewards_writer = list()
            
        return (processed_local_obs, processed_global_obs, np.array(self.edge_cell_mask), processed_veh_obs, processed_veh_mask), rewards, truncateds, dones, infos
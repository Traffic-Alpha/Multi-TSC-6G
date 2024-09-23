'''
Author: Maonan Wang
Date: 2024-09-23 15:19:31
LastEditTime: 2024-09-23 15:45:12
LastEditors: Maonan Wang
Description: Webster Methods
FilePath: /Multi-TSC-6G/rule_based_policy/webster_realtime_policy.py
'''
import numpy as np

def webster_policy(traffic_state, junction_phase_group, junction_movement_ids, max_occ_threshold:float=0.3, *args, **kwargs):
    # 计算当前每个 junction 所在的 phase
    current_junction_phase = {junction_id:0 for junction_id in traffic_state.keys()} # 记录当前路口的 phase index
    for junction_id, junction_info in traffic_state.items():
        this_phase = junction_info[-1][:,-1] # 这个路口此时哪些 movement 可以同行
        this_phase_movement_id = np.where(this_phase == 1)[0] # 当前 phase 对应的 movement id
        this_phase_movement = [junction_movement_ids[junction_id][i] for i in this_phase_movement_id] # 当前 phase 对应的 movement
        for phase_index, phase_movements in junction_phase_group[junction_id].items():
            if set(phase_movements) == set(this_phase_movement):
                current_junction_phase[junction_id] = phase_index
                break
        
    # 计算每一个 phase 的最大占有率
    junction_phase_max_occ = {junction_id:{} for junction_id in traffic_state.keys()}
    for junction_id, junction_phase_group in junction_phase_group.items():
        for phase_id, phase_movements in junction_phase_group.items(): # 统计每一个相位的最堵车的情况
            max_occ = 0
            for movement_id in phase_movements:
                _movement_index = junction_movement_ids[junction_id].index(movement_id)
                max_occ = max(max_occ, traffic_state[junction_id][-1][_movement_index][0])
            junction_phase_max_occ[junction_id][phase_id] = max_occ
    
    # 选择最堵车的 phase 作为 index
    action = {}
    for junction_id, current_phase_index in current_junction_phase.items():
        if junction_phase_max_occ[junction_id][current_phase_index] > max_occ_threshold:
            action[junction_id] = current_phase_index
        else:
            action[junction_id] = (current_phase_index+1)%4
        
    return action
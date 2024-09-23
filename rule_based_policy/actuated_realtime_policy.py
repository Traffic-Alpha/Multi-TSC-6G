'''
Author: Maonan Wang
Date: 2024-09-23 14:25:27
LastEditTime: 2024-09-23 15:12:07
LastEditors: Maonan Wang
Description: Actuated Methods
FilePath: /Multi-TSC-6G/rule_based_policy/actuated_realtime_policy.py
'''
def actuated_policy(traffic_state, junction_phase_group, junction_movement_ids, *args, **kwargs):
    # 首先计算每一个 phase 的最大占有率
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
    for key, sub_dict in junction_phase_max_occ.items():
        max_key = max(sub_dict, key=sub_dict.get)  # 找到最大值对应的键
        action[key] = max_key
        
    return action
'''
Author: Maonan Wang
Date: 2024-09-23 13:36:24
LastEditTime: 2024-09-23 15:46:58
LastEditors: Maonan Wang
Description: 固定配时控制策略
FilePath: /Multi-TSC-6G/rule_based_policy/fixed_time_policy.py
'''
PHASE_INDEX = 0  # 全局变量，控制 phase 的索引
CALL_COUNT = 0  # 全局变量，跟踪调用次数

def ft_policy(traffic_state, phase_duration:int=2, *args, **kwargs):
    global PHASE_INDEX, CALL_COUNT  # 声明全局变量
    action = {junction_id: PHASE_INDEX for junction_id in traffic_state.keys()}
    
    # 增加调用次数
    CALL_COUNT += 1
    
    # 根据 phase_duration 更新 PHASE_INDEX
    if CALL_COUNT >= phase_duration:
        PHASE_INDEX = (PHASE_INDEX + 1) % 4
        CALL_COUNT = 0  # 重置调用次数
    
    return action
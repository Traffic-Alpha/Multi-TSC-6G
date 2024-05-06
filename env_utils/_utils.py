'''
@Author: WANG Maonan
@Date: 2024-04-10 15:42:31
@Description: 一些环境创建使用到的工具
@LastEditTime: 2024-05-06 20:45:46
'''
import numpy as np
from collections import deque
from typing import List, Tuple, Dict

class TimeSeriesData:
    def __init__(self, N:int=10) -> None:
        self.N = N # 最多保存 N 个时间步骤的数据
        self.data = deque(maxlen=N)

    def add_data_point(self, timestamp:int, data):
        """记录每一个时间步的数据

        Args:
            timestamp (int): 仿真时间
            data: 需要保存的数据
        """
        # If deque is full, the oldest data will be removed automatically
        if len(self.data) == self.N:
            self.data.popleft()
        self.data.append((timestamp, data))  # Append new data as a tuple

    def get_recent_k_data(self, K: int=5):
        """Return the last K data points

        Args:
            K (int, optional): 返回的 K 个数据. Defaults to 5.
        """
        recent_data = list(self.data)[-K:]  # Get the last K data points
        return recent_data
    
    def get_data_point(self, timestamp:int):
        """返回指定时间步的数据 (这个的效率不高, 但是只在绘图的时候才会用到)
        """
        for ts, data in self.data:
            if ts == timestamp:
                return data
        return None  # Return None if timestamp not found

    def get_all_data(self):
        """返回所有的数据
        """
        return list(self.data)

    def get_time_stamps(self):
        """获得所有 key, 也就是所有的时间
        """
        return [ts for ts, _ in self.data]

    def calculate_edge_attribute(self, edge_id:str, attribute:str='vehicles'):
        """统计一个 edge 的结果, 一个 edge 包含多个 cell

        Args:
            edge_id (str): 需要计算的 edge id
            attribute (str, optional): 需要计算的属性. Defaults to 'vehicles'.

        Returns:
            _type_: _description_
        """
        attribute_timeseries = [] # 将所有时刻的属性累计起来
        for timestamp, data in self.data: # 遍历所有时间步, 这里需要按照时间顺序, 也就是按照数字大小
            _edge_attribute = [] # 统计某个时刻的属性
            for cell_info in data[edge_id]: # 遍历一个 edge 所有 cell 的数据
                _edge_attribute.append(cell_info.get(attribute, 0))
            attribute_timeseries.append(_edge_attribute)
        return attribute_timeseries

def merge_local_data(data:Tuple[int, Dict[str, List]]):
    """将每个时刻的 local data 进行合并

    Args:
        data (Tuple[int, Dict[str, List]]): 每个时刻, 每个路口的数据, 下面是一个例子:
            [
                (1, {'int1': [[1,2,3],[4,5,6]], 'int2': [[7,8,9], [1,2,3]]}),
                (2, {'int1': [[1,2,3],[4,5,6]], 'int2': [[7,8,9], [1,2,3]]}),
                ...
            ]

    Returns: 最后返回每个 tls 的数据, 输出例子如下所示:
    {
        'int1': [], # 多了一个时间维度
        'int2': ...
    }
    """
    # Initialize the result dictionary with IDs as keys and a list to hold arrays for each time step
    result = {id_key: [] for _, id_data in data for id_key in id_data}
    
    # Iterate over the input data
    for time, id_data in data:
        for id_key, _array in id_data.items(): # 每个 id 的数据
            result[id_key].append(_array)
    
    # stack
    for id_key in result.keys():
        result[id_key] = np.stack(result[id_key])
    
    return result

def direction_to_flags(direction):
    """
    Convert a direction string to a list of flags indicating the direction.

    :param direction: A string representing the direction (e.g., 's' for straight).
    :return: A list of flags for straight, left, and right.
    """
    return [
        1 if direction == 's' else 0,
        1 if direction == 'l' else 0,
        1 if direction == 'r' else 0
    ]

def one_hot_encode(input_list, target):
    # Initialize a list of zeros of length equal to the input list
    one_hot_vector = [0]*len(input_list)
    
    # If the target is in the input list, set the corresponding index to 1
    if target in input_list:
        one_hot_vector[input_list.index(target)] = 1
    
    return one_hot_vector

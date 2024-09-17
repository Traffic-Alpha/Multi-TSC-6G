<!--
 * @Author: WANG Maonan
 * @Date: 2024-04-09 21:28:52
 * @Description: Multi-Agent Traffic Signal Control under 6G
 * @LastEditTime: 2024-09-16 16:56:41
-->

Multi Agents for Traffic Signal Control Based on Global and Local Info (Network as a Sensor)


## 环境设计

tsc_env.TSCEnvironment 最核心的环境
global_local_wrapper.GlobalLocalInfoWrapper 对 state 和 reward 进行处理
pz_env.TSCEnvironmentPZ, 将环境转换为 petting zoo 对应的环境
torchrl_pz_wrapper.PettingZooWrapper，将环境转为为 torchrl 对应的环境


## 环境介绍
- Vehicle State，每个车辆的特征。这里一共是 100,如果数量不够 100,就填充 0,否则取 100. 每一个路口都是前 100 个和这个路口最接近的车辆
  - 车辆的速度
  - 车辆所在的 road, 使用 one-hot
  - 车辆所在的 lane position
  - 车辆的 waiting time
  - 车辆的 accumulated_waiting_time
  - 车辆记录路口的距离
- Local State，每个路口的局部特征。一个路口有 12 个方向，所以每个路口特征大小是 (12, 7)
  - 动态信息: occupancy, mean_speed
  - 车道静态信息: direction_flags（三种方向, one-hot）, lane_numbers
  - 信号灯信息: is_now_phase (动态信息)
- Global State，路网的全局信息。将 edge 且分为 cell，每个 cell 包含三个信息
  - vehicles，cell 内车辆数
  - average_waiting_time，cell 内车辆的平均等待时间
  - average_speed，cell 内车辆的平均速度


## 配置文件

- 环境配置文件
  - sumocfg
  - sumonet
  - 环境的仿真时间
  - Junction IDs (列表)
  - Action Space for each junction (字典)
- 实验配置文件（实验中，需要选择测试环境和模型名称）
  - 实验名称
  - 环境名称，读取环境的配置文件
  - 模型的名称
  
## 开始训练

- global 信息: 每一个路网 global info 信息的维度是会变化的，我们需要在配置文件中指定这个全局分割的大小等
- vehicle 信息: vehicle 最后一个所在的 lane id, 这里会因为 network 不一样导致维度不一样
- 上面的修改还需要修改 reset 部分返回的大小
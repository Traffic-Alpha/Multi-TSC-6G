<!--
 * @Author: WANG Maonan
 * @Date: 2024-04-09 21:28:52
 * @Description: Multi-Agent Traffic Signal Control under 6G
 * @LastEditTime: 2024-04-25 16:54:32
-->

Multi Agents for Traffic Signal Control Based on Global and Local Info (Network as a Sensor)


## 环境设计

tsc_env.TSCEnvironment 最核心的环境
global_local_wrapper.GlobalLocalInfoWrapper 对 state 和 reward 进行处理
pz_env.TSCEnvironmentPZ, 将环境转换为 petting zoo 对应的环境
torchrl_pz_wrapper.PettingZooWrapper，将环境转为为 torchrl 对应的环境


## 环境介绍

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
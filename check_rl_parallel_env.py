'''
@Author: WANG Maonan
@Date: 2023-10-30 23:01:03
@Description: 检查同时开启多个仿真环境
@LastEditTime: 2024-04-15 21:07:05
'''
from loguru import logger

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from env_utils.make_multi_tsc_env import make_parallel_env

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), file_log_level="INFO", terminal_log_level="WARNING")

if __name__ == '__main__':
    sumo_cfg = path_convert("./sumo_nets/3_ints/env/three_junctions.sumocfg")
    net_file = path_convert("./sumo_nets/3_ints/env/three_junctions.net.xml")
    log_path = path_convert('./log/3_ints')
    tsc_env = make_parallel_env(
        num_envs=6,
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        num_seconds=1300,
        tls_ids=['J1', 'J2', 'J3'],
        use_gui=False,
        log_file=log_path
    )
    rollouts = tsc_env.rollout(1_000, break_when_any_done=False)
    for r in rollouts:
        logger.info(f'RL: {r}')
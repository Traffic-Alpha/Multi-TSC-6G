'''
Author: Maonan Wang
Date: 2024-10-02 16:21:22
LastEditTime: 2024-10-02 17:14:14
LastEditors: Maonan Wang
Description: 分析 tripinfo 文件, 获得不同 policy 的指标
FilePath: /Multi-TSC-6G/analysis_tripinfo.py
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.analysis_output.tripinfo_analysis import TripInfoAnalysis

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level="INFO")

tripinfo_file = current_file_path("./trip_info.xml")
tripinfo_parser = TripInfoAnalysis(tripinfo_file)
tripinfo_parser.print_stats_as_table()
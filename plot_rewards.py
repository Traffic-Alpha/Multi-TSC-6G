'''
@Author: WANG Maonan
@Date: 2024-04-15 21:58:59
@Description: 绘制 Reward Curve with Standard Deviation
@LastEditTime: 2024-04-25 22:15:40
'''
from tshub.utils.plot_reward_curves import plot_reward_curve, plot_multi_reward_curves
from tshub.utils.get_abs_path import get_abs_path
path_convert = get_abs_path(__file__)


if __name__ == '__main__':
    log_labels = {
        "Exp1": [path_convert(f'./log/Exp1/{i}.monitor.csv')for i in range(16)],
        "Exp2": [path_convert(f'./log/Exp2/{i}.monitor.csv')for i in range(16)]
    }
    output_file = path_convert('./reward.png')
    plot_multi_reward_curves(log_labels, output_file, window_size=3)
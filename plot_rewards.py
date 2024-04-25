'''
@Author: WANG Maonan
@Date: 2024-04-15 21:58:59
@Description: 绘制 Reward Curve with Standard Deviation
@LastEditTime: 2024-04-25 10:59:26
'''
from tshub.utils.plot_reward_curves import plot_reward_curve
from tshub.utils.get_abs_path import get_abs_path
path_convert = get_abs_path(__file__)


if __name__ == '__main__':
    log_files = [
        path_convert(f'./log/train_mixed/{i}.monitor.csv')
        for i in range(12)
    ]
    output_file = path_convert('./reward.png')
    plot_reward_curve(log_files, output_file, window_size=3, fill_outliers=False)
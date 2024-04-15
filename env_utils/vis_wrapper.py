'''
@Author: WANG Maonan
@Date: 2024-04-09 22:33:35
@Description: 根据 global info 来绘制图像, 这里按照 edge 进行绘制
如果要绘制图像, 首先统计每个 lane 的值, 接着根据 lane 的 shape 进行绘制即可
@LastEditTime: 2024-04-10 16:20:12
'''
import numpy as np
import gymnasium as gym
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class VisWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def reset(self, seed: int | None = None):
        return self.env.reset(seed=seed)
    
    def step(self, action):
        return self.env.step(action)

    def __aggregate_statistics(self, timestamp):
        aggregated_statistics = {}
        for edge_id, cells in self.env.edge_cells_timeseries.get_data_point(timestamp).items():
            total_vehicles = sum(cell['vehicles'] for cell in cells)
            total_waiting_time = sum(cell['total_waiting_time'] for cell in cells)
            total_speed = sum(cell['total_speed'] for cell in cells)
            total_co2_emission = sum(cell['total_co2_emission'] for cell in cells)
            
            # 统计包含车辆的 cell 个数
            cells_with_vehicles = sum(1 for cell in cells if cell['vehicles'] > 0)
            
            average_waiting_time = (total_waiting_time / cells_with_vehicles) if cells_with_vehicles else 0
            average_speed = (total_speed / cells_with_vehicles) if cells_with_vehicles else 0
            average_co2_emission = (total_co2_emission / cells_with_vehicles) if cells_with_vehicles else 0
            
            aggregated_statistics[edge_id] = {
                'total_vehicles': total_vehicles,
                'total_waiting_time': total_waiting_time,
                'average_waiting_time': average_waiting_time,
                'total_speed': total_speed,
                'average_speed': average_speed,
                'total_co2_emission': total_co2_emission,
                'average_co2_emission': average_co2_emission
            }
        return aggregated_statistics
    
    def plot_map(self, timestamp, attributes=['total_vehicles', ], is_plot_edge=False):
        aggregated_statistics = self.__aggregate_statistics(timestamp) # 这里统计每一个 edge 的信息
        # Define a colormap
        cmap = plt.cm.GnBu
        
        for attribute in attributes:
            # Normalize the attribute values for color mapping
            attr_values = [aggregated_statistics[edge_id][attribute] for edge_id in aggregated_statistics]
            norm = plt.Normalize(vmin=min(attr_values), vmax=max(attr_values), clip=True)
            
            fig, ax = plt.subplots()
            # 绘制 node
            for _, node_info in self.env.node_infos.items():
                node_shape = np.array(node_info.get('shape'))
                x, y = node_shape[:, 0], node_shape[:, 1]
                ax.plot(x, y, color='gray', linewidth=1) 

            # 绘制 lane
            for _, lane_info in self.env.lane_infos.items():
                edge_id = lane_info['edge_id'] # 获得 lane 对应的 edge id
                if edge_id in aggregated_statistics:
                    # Get the attribute value for the current edge
                    attr_value = aggregated_statistics[edge_id][attribute]
                    # Normalize the attribute value
                    normalized_value = norm(attr_value)
                    # Get the color from the colormap
                    color = cmap(normalized_value)
                    
                    # Get the shape for the current edge
                    shape = np.array(lane_info['shape'])
                    
                    # Plot the edge shape
                    if is_plot_edge:
                        ax.fill(*zip(*shape), color=color, edgecolor='black')  # Add edgecolor here
                    else: # 不绘制边框
                        ax.fill(*zip(*shape), color=color)
                        
            # Create a ScalarMappable and use it to create the colorbar
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # Set a dummy array for the ScalarMappable.
            fig.colorbar(sm, ax=ax)
            ax.set_title(f'Map Visualization for {attribute}')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.axis('equal')

            # Show the figure
            plt.show()
    
    def plot_edge_attribute(self, edge_id:str, attribute:str='vehicles'):
        """
        Plot the attribute over time for each cell in an edge.

        :param data: A 2D list where each sublist represents the congestion at each cell for a given time.
        """
        # Convert the data to a numpy array for better handling
        data = self.env.edge_cells_timeseries.calculate_edge_attribute(edge_id, attribute)
        data_array = np.array(data)

        # Create a meshgrid for plotting
        time = np.arange(data_array.shape[0] + 1)
        cell_index = np.arange(data_array.shape[1] + 1)
        T, C = np.meshgrid(time, cell_index)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(T, C, data_array.T, shading='auto')  # Transpose data_array to align with the meshgrid

        # Set the labels and title
        plt.xlabel('Time')
        plt.ylabel('Cell Index')
        plt.title('Congestion Level Over Time')

        # Show the plot with a color bar
        plt.colorbar(label='Congestion Level')
        plt.show()
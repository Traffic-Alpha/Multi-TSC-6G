'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:48
@Description: Actor Network, Local Networks for decision
LastEditTime: 2025-10-29 21:59:58
'''
import torch
from torch import nn
import torch.nn.functional as F

class LocalFeatureProcessor(nn.Module):
    """局部特征处理器 - 处理每个路口的局部特征
    """
    def __init__(self, time_steps=5, num_directions=12, local_feat_dim=7, hidden_dim=64):
        super(LocalFeatureProcessor, self).__init__()
        self.time_steps = time_steps
        self.num_directions = num_directions
        self.local_feat_dim = local_feat_dim
        self.hidden_dim = hidden_dim
        
        # 时间维度自注意力
        self.time_attention = nn.MultiheadAttention(
            embed_dim=local_feat_dim,
            num_heads=1,
            batch_first=True
        )
        
        # 方向维度自注意力
        self.direction_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(time_steps * local_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, local_features):
        # local_features: [batch_size, num_intersections, time_steps, num_directions, local_feat_dim], 这里 bs 可以是同时开启的环境个数
        batch_size, num_intersections, time_steps, num_directions, feat_dim = local_features.shape
        
        # 重塑为 [batch_size * num_intersections * num_directions, time_steps, feat_dim]
        x = local_features.reshape(batch_size * num_intersections * num_directions, time_steps, feat_dim)
        
        # 1. 时间维度自注意力
        # [batch_size * num_intersections * num_directions, time_steps, feat_dim]
        x, _ = self.time_attention(x, x, x)
        
        # 展平时间维度, [batch_size * num_intersections * num_directions, time_steps * feat_dim]
        x = x.reshape(batch_size * num_intersections * num_directions, -1)
        
        # 特征编码, [batch_size * num_intersections * num_directions, hidden_dim]
        x = self.feature_encoder(x)
        
        # 2. 方向维度自注意力
        # 重塑为 [batch_size * num_intersections, num_directions, hidden_dim]
        x_reshaped = x.reshape(batch_size * num_intersections, num_directions, self.hidden_dim)
        x_attended, _ = self.direction_attention(x_reshaped, x_reshaped, x_reshaped)
        
        # 重塑回原始维度, [batch_size, num_intersections, num_directions, hidden_dim]
        x_attended = x_attended.reshape(batch_size, num_intersections, num_directions, self.hidden_dim)
        
        # 对方向维度进行平均池化，得到每个路口的聚合表示
        # [batch_size, num_intersections, hidden_dim]
        intersection_rep = torch.mean(x_attended, dim=2)  # [batch_size, num_intersections, hidden_dim]
        
        return intersection_rep

class ActorNetwork(nn.Module):
    def __init__(self, action_size):
        super(ActorNetwork, self).__init__()
        self.local_processor = LocalFeatureProcessor(
            time_steps=5,
            num_directions=12,
            local_feat_dim=7,
            hidden_dim=64
        )
        self.fc3 = nn.Linear(in_features=64, out_features=action_size)

    def forward(self, x):
        x = x['local'] # actor 只使用 local 信息, x['local']
        batch_size, num_intersections, time_steps, num_directions, feat_dim = x.shape

        x = self.local_processor(x)
        x = self.fc3(x)

        return x.view(batch_size, num_intersections, -1)
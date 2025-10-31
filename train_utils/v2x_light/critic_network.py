'''
@Author: WANG Maonan
@Date: 2024-04-25 16:27:56
@Description: Critic Network
LastEditTime: 2025-10-30 16:29:45
'''
import math
import torch
from torch import nn
import torch.nn.functional as F

# class MultiLayerTransformer(nn.Module):
#     """多层Transformer编码器"""
#     def __init__(self, hidden_dim=64, num_heads=4, num_layers=3, dropout=0.1):
#         super(MultiLayerTransformer, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
        
#         # Transformer编码器层
#         encoder_layers = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
#         # 层归一化
#         self.layer_norm = nn.LayerNorm(hidden_dim)
        
#     def forward(self, x, src_key_padding_mask=None):
#         # x: [batch_size, seq_len, hidden_dim]
#         # src_key_padding_mask: [batch_size, seq_len] - True表示要mask的位置
        
#         # 通过多层Transformer
#         output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
#         output = self.layer_norm(output)
        
#         return output

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 线性变换层
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.w_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用 mask（如果有）
        if mask is not None:
            # mask形状: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到value
        attention_output = torch.matmul(attention_weights, V)
        
        # 重塑回原始形状
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # 输出线性变换
        output = self.w_o(attention_output)
        
        return output

class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, hidden_dim=64, ff_dim=128, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, hidden_dim=64, num_heads=4, ff_dim=128, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # 自注意力机制
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(hidden_dim, ff_dim, dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力子层 + 残差连接 + 层归一化
        attention_output = self.self_attention(x, x, x, mask)
        x = self.attention_norm(x + self.dropout(attention_output))
        
        # 前馈网络子层 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + self.dropout(ff_output))
        
        return x

class MultiLayerTransformer(nn.Module):
    """多层Transformer编码器"""
    def __init__(self, hidden_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super(MultiLayerTransformer, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 创建多个Transformer编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim * 2,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, src_key_padding_mask=None):
        # x: [batch_size, seq_len, hidden_dim]
        # src_key_padding_mask: [batch_size, seq_len] - True表示要mask的位置
        
        # 逐层通过Transformer编码器
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        
        # 最终层归一化
        output = self.layer_norm(x)
        
        return output


class LocalFeatureProcessor(nn.Module):
    """局部特征处理器, 处理每个路口的局部特征"""
    def __init__(
            self, 
            time_steps=5, num_directions=12, 
            local_feat_dim=7, hidden_dim=64,
            num_time_layers=2, num_direction_layers=2
        ):
        super(LocalFeatureProcessor, self).__init__()
        self.time_steps = time_steps
        self.num_directions = num_directions
        self.local_feat_dim = local_feat_dim
        self.hidden_dim = hidden_dim
        
        # 多层时间维度Transformer
        self.time_transformer = MultiLayerTransformer(
            hidden_dim=local_feat_dim,
            num_heads=1,
            num_layers=num_time_layers
        )
        
        # 多层方向维度Transformer
        self.direction_transformer = MultiLayerTransformer(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=num_direction_layers
        )
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(time_steps * local_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, local_features):
        # local_features: [parallel_envs, batch_size, num_intersections, time_steps, num_directions, local_feat_dim]
        parallel_envs, batch_size, num_intersections, time_steps, num_directions, feat_dim = local_features.shape
        
        # 重塑为 [parallel_envs * batch_size * num_intersections * num_directions, time_steps, feat_dim]
        x = local_features.reshape(parallel_envs * batch_size * num_intersections * num_directions, time_steps, feat_dim)
        
        # 多层时间维度Transformer
        x = self.time_transformer(x)
        
        # 展平时间维度
        # [parallel_envs * batch_size * num_intersections * num_directions, time_steps * feat_dim]
        x = x.reshape(parallel_envs * batch_size * num_intersections * num_directions, -1)
        
        # 特征编码
        # [parallel_envs * batch_size * num_intersections * num_directions, hidden_dim]
        x = self.feature_encoder(x)
        
        # 直接重塑为 [parallel_envs * batch_size * num_intersections, num_directions, hidden_dim] 进行方向注意力
        x_reshaped = x.reshape(parallel_envs * batch_size * num_intersections, num_directions, self.hidden_dim)
        x_attended = self.direction_transformer(x_reshaped)
        
        # 重塑为 [parallel_envs, batch_size, num_intersections, num_directions, hidden_dim]
        x_attended = x_attended.reshape(parallel_envs, batch_size, num_intersections, num_directions, self.hidden_dim)
        
        # 对方向维度进行平均池化，得到每个路口的聚合表示
        intersection_rep = torch.mean(x_attended, dim=3)  # [parallel_envs, batch_size, num_intersections, hidden_dim]
        
        return intersection_rep, x_attended
    

class GlobalFeatureProcessor(nn.Module):
    """全局特征处理器, 处理全局道路网络特征"""
    def __init__(
            self, 
            time_steps=5, num_cells=11,
            global_feat_dim=3, hidden_dim=64,
            num_time_layers=2, num_cell_layers=2
        ):
        super(GlobalFeatureProcessor, self).__init__()
        self.time_steps = time_steps
        self.num_cells = num_cells
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        
        # 多层时间维度Transformer
        self.time_transformer = MultiLayerTransformer(
            hidden_dim=global_feat_dim,
            num_heads=1,
            num_layers=num_time_layers
        )

        # 时间特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # 多层空间Transformer（cell级别）
        self.cell_transformer = MultiLayerTransformer(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=num_cell_layers
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, global_features, global_mask):
        # global_features: [parallel_envs, batch_size, num_edges, time_steps, num_cells, global_feat_dim]
        # global_mask: [parallel_envs, batch_size, num_edges, num_cells]
        parallel_envs, batch_size, num_edges, time_steps, num_cells, feat_dim = global_features.shape
        
        # 首先处理时间维度：对每个 cell 的每个时间步进行编码
        # 重塑为 [parallel_envs * batch_size * num_edges * num_cells, time_steps, feat_dim]
        x = global_features.reshape(parallel_envs * batch_size * num_edges * num_cells, time_steps, feat_dim)
        
        # 多层时间维度Transformer
        x_encoded = self.time_transformer(x)
        
        # 对时间维度进行平均池化, 去除了时间维度
        # [parallel_envs * batch_size * num_edges * num_cells, hidden_dim]
        x_time_pooled = torch.mean(x_encoded, dim=1)
        x_time_pooled = self.feature_encoder(x_time_pooled)
        
        # 重塑为 [parallel_envs * batch_size * num_edges, num_cells, hidden_dim] 进行空间注意力
        x_for_attention = x_time_pooled.reshape(parallel_envs * batch_size * num_edges, num_cells, self.hidden_dim)
        
        # 为注意力机制创建mask（True表示要mask掉的位置）
        # global_mask中1表示有效，0表示无效，所以需要反转
        attention_mask = ~global_mask.reshape(parallel_envs * batch_size * num_edges, num_cells).bool()
        
        # 多层空间Transformer [parallel_envs * batch_size * num_edges, num_cells, hidden_dim]
        # 使用src_key_padding_mask来处理可变长度的序列
        x_attended = self.cell_transformer(x_for_attention, src_key_padding_mask=attention_mask)
        
        # 对有效的cell进行平均池化得到每个edge的表示
        # 计算每个edge的有效cell数量
        valid_cells = global_mask.reshape(parallel_envs * batch_size * num_edges, num_cells).sum(dim=1, keepdim=True)  # [parallel_envs*batch_size*num_edges, 1]
        
        # 应用 mask：将无效位置置为 0
        mask_expanded = global_mask.reshape(parallel_envs * batch_size * num_edges, num_cells, 1).expand(-1, -1, self.hidden_dim)
        x_masked = x_attended * mask_expanded.float()
        
        # 对有效的cell进行平均池化
        edge_rep = x_masked.sum(dim=1) / (valid_cells + 1e-8)  # [parallel_envs*batch_size*num_edges, hidden_dim]
        
        # 包含每个 edge 的信息, [parallel_envs, batch_size, num_edges, hidden_dim]
        edge_rep = self.output_proj(edge_rep)
        edge_rep = edge_rep.reshape(parallel_envs, batch_size, num_edges, self.hidden_dim)
        
        return edge_rep

class MultiLayerCrossAttention(nn.Module):
    """多层交叉注意力模块"""
    def __init__(self, hidden_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super(MultiLayerCrossAttention, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, local_queries, global_keys_values, key_padding_mask=None):
        # local_queries: [batch_size, num_queries, hidden_dim]
        # global_keys_values: [batch_size, num_kv, hidden_dim]
        # key_padding_mask: [batch_size, num_kv] - True表示要mask的位置
        
        x = local_queries
        attention_weights_list = []
        
        for layer in self.layers:
            x, attention_weights = layer(x, global_keys_values, key_padding_mask)
            attention_weights_list.append(attention_weights)
        
        return x, attention_weights_list

class CrossAttentionLayer(nn.Module):
    """单层交叉注意力层"""
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, local_queries, global_keys_values, key_padding_mask=None):
        # 交叉注意力：使用局部信息作为 query，全局信息作为 key 和 value
        attended_local, attention_weights = self.cross_attention(
            query=local_queries,
            key=global_keys_values,
            value=global_keys_values,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接和层归一化
        x = self.layer_norm1(local_queries + self.dropout(attended_local))
        
        # 前馈网络
        ff_out = self.feed_forward(x)
        
        # 残差连接和层归一化
        output = self.layer_norm2(x + self.dropout(ff_out))
        
        return output, attention_weights

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块 - 使用全局信息增强局部信息，支持多层Transformer"""
    def __init__(self, hidden_dim=64, num_heads=4, num_layers=3):
        super(CrossAttentionFusion, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 多层交叉注意力
        self.multi_layer_cross_attention = MultiLayerCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
    def forward(self, local_queries, global_keys_values, global_mask=None):
        # local_queries: [parallel_envs, batch_size, num_intersections, hidden_dim]
        # global_keys_values: [parallel_envs, batch_size, num_edges, hidden_dim]
        # global_mask: [parallel_envs, batch_size, num_edges] - 如果有的话
        
        parallel_envs, batch_size, num_intersections, hidden_dim = local_queries.shape
        parallel_envs, batch_size, num_edges, hidden_dim = global_keys_values.shape

        # 重塑为batch first
        local_queries_bf = local_queries.reshape(parallel_envs * batch_size, num_intersections, hidden_dim)
        global_keys_values_bf = global_keys_values.reshape(parallel_envs * batch_size, num_edges, hidden_dim)
        
        # 准备key_padding_mask（如果需要）
        key_padding_mask = None
        if global_mask is not None:
            # global_mask: [parallel_envs, batch_size, num_edges]
            key_padding_mask = ~global_mask.reshape(parallel_envs * batch_size, num_edges).bool()
        
        # 多层交叉注意力
        output, attention_weights_list = self.multi_layer_cross_attention(
            local_queries_bf, global_keys_values_bf, key_padding_mask
        )
        
        # 还原形状
        output = output.reshape(parallel_envs, batch_size, num_intersections, hidden_dim)

        return output, attention_weights_list

class CriticNetwork(nn.Module):
    def __init__(self,
            time_steps=5,
            num_directions=12,
            local_feat_dim=7,
            num_cells=11,
            global_feat_dim=3,
            hidden_dim=64,
            num_time_layers=2,
            num_direction_layers=2,
            num_cell_layers=2,
            num_fusion_layers=3
        ):
        super(CriticNetwork, self).__init__()
        
        # 局部特征处理器
        self.local_processor = LocalFeatureProcessor(
            time_steps=time_steps,
            num_directions=num_directions,
            local_feat_dim=local_feat_dim,
            hidden_dim=hidden_dim,
            num_time_layers=num_time_layers,
            num_direction_layers=num_direction_layers
        )
        
        # 全局特征处理器
        self.global_processor = GlobalFeatureProcessor(
            time_steps=time_steps,
            num_cells=num_cells,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_time_layers=num_time_layers,
            num_cell_layers=num_cell_layers
        )
        
        # 交叉注意力融合
        self.cross_fusion = CrossAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=num_fusion_layers
        )
        
        # 价值头 - 输出维度为1
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        local_features, global_features, global_mask = x['local'], x['global'], x['global_mask']
        
        # 检查输入是否有parallel_envs维度，如果没有则添加一个虚拟维度
        added_parallel_envs = False
        if len(local_features.shape) == 5:  # 没有 parallel_envs 维度
            # 添加一个parallel_envs维度
            local_features = local_features.unsqueeze(0)
            global_features = global_features.unsqueeze(0)
            global_mask = global_mask.unsqueeze(0)
            added_parallel_envs = True
        
        # 处理全局特征 - 这里每个路口信息一样, 使用一个信息即可
        global_features = global_features.select(2, 0)  # 选择num_edges维度的第一个
        global_mask = global_mask.select(2, 0)
        
        # 处理局部特征
        # intersection_rep: [parallel_envs, batch_size, num_intersections, hidden_dim]
        intersection_rep, direction_rep = self.local_processor(local_features)
        
        # 处理全局特征
        # edge_rep: [parallel_envs, batch_size, num_edges, hidden_dim]
        edge_rep = self.global_processor(global_features, global_mask)
        
        # 交叉注意力融合
        # fused_intersection_rep: [parallel_envs, batch_size, num_intersections, hidden_dim]
        fused_intersection_rep, attention_weights = self.cross_fusion(
            intersection_rep, edge_rep, None
        )
        
        # 应用价值头，将最后一个维度从hidden_dim变为1
        value_output = self.value_head(fused_intersection_rep)
        
        # 如果之前添加了虚拟的 parallel_envs 维度，现在移除它
        if added_parallel_envs:
            value_output = value_output.squeeze(0)
        
        return value_output
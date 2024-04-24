import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from .layers import Transformer, CrossTransformer

class TransformerModule(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerModule, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        return self.encoder(x)

class CriticTransformer_local(nn.Module):
    def __init__(self):
        super(CriticTransformer_local, self).__init__()
        self.fc1 = nn.Linear(7, 32)
        self.movement_transformer = TransformerModule(32, 8, 4, 32)
        self.pooling = nn.AdaptiveAvgPool1d(32)
        self.time_transformer = TransformerModule(32, 8, 4, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, local_global_x):
        x_local = local_global_x['local']       # 5, 600, 3,  5, 12,  7
        x_global = local_global_x['global']     # 5, 600, 3, 20,  5, 11, 3
        # n_envs, batchsize, n_agents = list(x_local.shape[:-3])
        env_batch_nagents = list(x_local.shape[:-3])    # 包含 n_envs, batchsize 和 n_agents
        timeseries, movement, feature_num = x_local.shape[-3:]
        x_local = x_local.view(-1, timeseries, movement, feature_num)
        env_batch_nagents_size = x_local.shape[0]
        x_local = x_local.view(-1, movement, feature_num)
        x_local = self.fc1(x_local)
        x_local = self.movement_transformer(x_local)
        x_local = x_local.view(env_batch_nagents_size, timeseries, -1)
        x_local = self.pooling(x_local)
        x_local = self.time_transformer(x_local)
        x_local = x_local.view(env_batch_nagents_size, -1)
        x_local = self.pooling(x_local)
        x_local = self.fc2(x_local)
        return x_local.view(*env_batch_nagents, -1)

class CriticTransformer_fused(nn.Module):
    def __init__(self, depth=1, encoder_depth=2, dim=32, heads=8, dim_head=64, mlp_dim=64, dropout=0.1):
        super(CriticTransformer_fused, self).__init__()
        self.depth = depth
        self.local_fc = nn.Sequential(nn.Linear(7, dim),
                                      nn.ReLU(),)
        self.global_fc = nn.Sequential(nn.Linear(60, dim),
                                      nn.ReLU(),)
        self.local_time_SAs = nn.ModuleList([])
        self.global_spatial_SAs = nn.ModuleList([])
        self.global_time_SAs = nn.ModuleList([])
        self.fuse_CAs = nn.ModuleList([])
        for i in range(self.depth):
            self.local_time_SAs.append(Transformer(dim=dim, depth=encoder_depth, heads=heads, dim_head=dim_head,
                                                   mlp_dim=mlp_dim, dropout=dropout))
            self.global_spatial_SAs.append(Transformer(dim=dim, depth=encoder_depth, heads=heads, dim_head=dim_head,
                                                   mlp_dim=mlp_dim, dropout=dropout))
            self.global_time_SAs.append(Transformer(dim=dim, depth=encoder_depth, heads=heads, dim_head=dim_head,
                                                   mlp_dim=mlp_dim, dropout=dropout))
            self.fuse_CAs.append(CrossTransformer(dim=dim, depth=encoder_depth, heads=heads, dim_head=dim_head,
                                                   mlp_dim=mlp_dim, dropout=dropout))
        self.pooling = nn.AdaptiveAvgPool1d(dim)
        self.conv = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)
        self.fuse_fc = nn.Linear(dim, 1)

    def forward(self, local_global_x):
        x_local = local_global_x['local']       # 5, 600, 3,  5, 12,  7
        x_global = local_global_x['global']     # 5, 600, 3, 20,  5, 11, 3
        if x_local.ndim == 6:
            nba = list(x_local.shape[:3])
            x_local = rearrange(x_local, 'n b a ... -> (n b a) ...')
            x_global = rearrange(x_global, 'n b a ... -> (n b a) ...')
        else:
            nba = list(x_local.shape[:2])
            x_local = rearrange(x_local, 'nb a ... -> (nb a) ...')      # x_local: nba, time=5, m=12, d=7
            x_global = rearrange(x_global, 'nb a ... -> (nb a ) ...')   # x_global: nba, slice=20 time=5, place=11, d=3

        time, place = x_global.shape[2: 4]

        # prep local
        x_local = self.local_fc(x_local)        # x_local: nba, t=5, m=12, dim=64
        x_local = rearrange(x_local, '... m d -> ... (m d)')
        x_local = self.pooling(x_local)         # x_local: nba, t=5, dim=64

        # prep global
        x_global = rearrange(x_global, 'nba s t p d -> nba t p (s d)')  # x_global: nba, time=5, place, slice*dim=33
        x_global = self.global_fc(x_global)     # x_local: nba, t=5, p=11, dim=64
        for i in range(self.depth):
            # local branch
            x_local = self.local_time_SAs[i](x_local)

            # global branch
            x_global = rearrange(x_global, 'nba t p d -> (nba p) t d')
            x_global = self.global_time_SAs[i](x_global)
            x_global = rearrange(x_global, '(nba p) t d -> (nba t) p d', p=place)
            x_global = self.global_spatial_SAs[i](x_global)

            # fuse branch
            x_local_query = rearrange(x_local, 'nba t d -> (nba t) d')
            x_local_query = repeat(x_local_query, 'nbat d -> nbat p d', p=place)
            x_global = self.fuse_CAs[i](x_local_query, x_global)    # nbat p d
            x_global = rearrange(x_global, '(nba t) p d -> nba t p d', t=time)

        x_global = rearrange(x_global, 'nba t p d -> nba t (p d)')
        x_global = self.pooling(x_global)
        # x_global = rearrange(x_global, 'nba t d -> nba d t')
        x_global = self.conv(x_global).squeeze(1)
        x_global = self.fuse_fc(x_global)
        return x_global.view(*nba, -1)
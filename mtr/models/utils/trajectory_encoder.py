import torch
import torch.nn as nn
from ..utils import common_layers
from pytorch_tcn import TCN
from isab_pytorch import ISAB

class TrajectoryEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=3, out_channels=None, time_encoder='rnn', kernel_size=3, nhead=4):
        super().__init__()
        """
        Args: time_enocder: 'rnn', 'transformer', 'tcn'
        """
        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=True
        )
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.nhead = nhead
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.time_encoder = time_encoder
        if time_encoder == 'rnn':
            self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif time_encoder == 'transformer':
            transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        elif time_encoder == 'tcn':
            self.encoder = TCN(num_inputs=hidden_dim, num_channels=[hidden_dim] * num_layers, kernel_size=kernel_size, input_shape='NLC')
        elif time_encoder == 'set_transformer':
            self.encoder = ISAB(dim=hidden_dim, heads=nhead, num_latents=1, latent_self_attend=True)
        else:
            raise ValueError(f"Invalid time_encoder: {time_encoder}")
        
        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, trajectories, trajectories_mask):
        """
        Args:
            trajectories (num_center_agents, num_agents, num_timesteps, C):
            trajectories_mask (num_center_agents, num_agents, num_timesteps):

        Returns:
        """
        num_center_agents, num_agents,  num_timesteps, C = trajectories.shape

        # pre-mlp
        trajectories_feature_valid = self.pre_mlps(trajectories[trajectories_mask])  # (N, C)
        trajectories_feature = trajectories.new_zeros(num_center_agents, num_agents,  num_timesteps, trajectories_feature_valid.shape[-1])
        trajectories_feature[trajectories_mask] = trajectories_feature_valid

        # # get global feature
        # pooled_feature = polylines_feature.max(dim=2)[0]
        # polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # # mlp
        # polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        # feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        # feature_buffers[polylines_mask] = polylines_feature_valid

        # # max-pooling
        # feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)

        # Encode timeseries
        if self.time_encoder == 'rnn':
            # Rearrange input dimensions to fit GRU
            trajectories_feature = trajectories_feature.view(-1, num_timesteps, trajectories_feature.shape[-1])
            _, hidden = self.encoder(trajectories_feature)
            feature_buffers = hidden[-1].view(num_center_agents, num_agents, -1)
        elif self.time_encoder == 'transformer':
            trajectories_feature = trajectories_feature.view(-1, num_timesteps, trajectories_feature.shape[-1])
            out = self.encoder(trajectories_feature)
            feature_buffers = out[:, -1, :].view(num_center_agents, num_agents, -1)
        elif self.time_encoder == 'tcn':
            # TODO: Check for output shapes between context and target
            pass
        elif self.time_encoder == 'set_transformer':
            trajectories_feature = trajectories_feature.view(-1, num_timesteps, trajectories_feature.shape[-1])
            out, _ = self.encoder(trajectories_feature)
            feature_buffers = out[:, -1, :].view(num_center_agents, num_agents, -1)
        
        # out-mlp 
        if self.out_mlps is not None:
            valid_mask = (trajectories_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(num_center_agents, num_agents, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers

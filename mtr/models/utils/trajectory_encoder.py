import torch
import torch.nn as nn
from ..utils import common_layers
from pytorch_tcn import TCN


class TrajectoryEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=3, out_channels=None, time_encoder='rnn', kernel_size=3):
        super().__init__()
        """
        Args: time_enocder: 'rnn', 'transformer', 'tcn'
        """
        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=True
        )

        self.time_encoder = time_encoder
        if time_encoder == 'rnn':
            self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif time_encoder == 'transformer':
            transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
            self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        elif time_encoder == 'tcn':
            self.encoder = TCN(num_inputs=hidden_dim, num_channels=[hidden_dim] * num_layers, kernel_size=kernel_size, input_shape='NLC')
        else:
            raise ValueError(f"Invalid time_encoder: {time_encoder}")
        
        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        batch_size, num_polylines,  num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers

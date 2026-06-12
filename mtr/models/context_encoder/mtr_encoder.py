# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import math
import torch
import torch.nn as nn
from torchvision.ops import MLP


from mtr.models.utils.transformer import transformer_encoder_layer, position_encoding_utils
from mtr.models.utils import polyline_encoder
from mtr.utils import common_utils
from mtr.ops.knn import knn_utils


class MTREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL
        )
        self.use_pose_encoder = self.model_cfg.get('USE_POSE_ENCODER', True)
        self.use_cross_attn_pose = self.model_cfg.get('USE_CROSS_ATTN_POSE', False)
        if self.use_pose_encoder:
            if self.use_cross_attn_pose:
                # H5: cross-attention pose encoder
                # Projects each pose frame to D_MODEL, then cross-attends from agent feature
                self.pose_proj = nn.Linear(144, self.model_cfg.D_MODEL)
                self.pose_cross_attn = nn.MultiheadAttention(
                    embed_dim=self.model_cfg.D_MODEL,
                    num_heads=self.model_cfg.NUM_ATTN_HEAD,
                    dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                    batch_first=True
                )
                # H5b: optional sinusoidal positional encoding for temporal ordering
                self.use_pose_sinusoidal_pe = self.model_cfg.get('USE_POSE_SINUSOIDAL_PE', False)
                if self.use_pose_sinusoidal_pe:
                    T = self.model_cfg.get('NUM_PAST_TIMESTAMPS', 11)
                    d = self.model_cfg.D_MODEL
                    pe = torch.zeros(T, d)
                    pos = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
                    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
                    pe[:, 0::2] = torch.sin(pos * div)
                    pe[:, 1::2] = torch.cos(pos * div)
                    self.register_buffer('pose_sinusoidal_pe', pe.unsqueeze(0))  # (1, T, D_MODEL)
            else:
                # Default: GRU pose encoder
                self.pose_encoder = torch.nn.GRU(
                    input_size=144,
                    hidden_size=self.model_cfg.D_MODEL,
                    num_layers=self.model_cfg.NUM_LAYER_IN_POSE_GRU,
                    dropout=self.model_cfg.get('DROPOUT_OF_POSE_GRU', 0.0),
                    batch_first=True
                )

            self.pose_fuser = MLP(
                in_channels=self.model_cfg.D_MODEL * 2,
                hidden_channels=[self.model_cfg.D_MODEL, self.model_cfg.D_MODEL],
                dropout=self.model_cfg.get('DROPOUT_OF_POSE_FUSER', 0.0),
            )
            self.pose_fuser_residual = self.model_cfg.get('POSE_FUSER_RESIDUAL', False)
            if self.pose_fuser_residual:
                # Zero-init the last Linear so pose_fuser ≈ 0 at init, preserving
                # pre-trained backbone features when loading from a trajectory-only ckpt.
                for m in reversed(list(self.pose_fuser.modules())):
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
                        break

        # build transformer encoder layers
        self.use_local_attn = self.model_cfg.get('USE_LOCAL_ATTN', False)
        self_attn_layers = []
        for _ in range(self.model_cfg.NUM_ATTN_LAYERS):
            self_attn_layers.append(self.build_transformer_encoder_layer(
                d_model=self.model_cfg.D_MODEL,
                nhead=self.model_cfg.NUM_ATTN_HEAD,
                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                normalize_before=False,
                use_local_attn=self.use_local_attn
            ))

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg.D_MODEL

    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder

    def build_transformer_encoder_layer(self, d_model, nhead, dropout=0.1, normalize_before=False, use_local_attn=False):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            normalize_before=normalize_before, use_local_attn=use_local_attn
        )
        return single_encoder_layer

    def apply_global_attn(self, x, x_mask, x_pos):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)
 
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask,
                pos=pos_embedding
            )
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out

    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        batch_offsets = common_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack,  batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        # positional encoding
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_stack[None, :, 0:2], hidden_dim=d_model)[0]

        # local attn
        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict['input_dict']
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].cuda(), input_dict['obj_trajs_mask'].cuda() 
        map_polylines, map_polylines_mask = input_dict['map_polylines'].cuda(), input_dict['map_polylines_mask'].cuda()

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].cuda()
        map_polylines_center = input_dict['map_polylines_center'].cuda()
        track_index_to_predict = input_dict['track_index_to_predict']

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (num_center_objects, num_polylines, C)

        # Apply Pose Encoder (optional — disabled for trajectory-only baseline)
        if self.use_pose_encoder:
            obj_poses, obj_poses_mask = input_dict['obj_poses'].cuda(), input_dict['obj_poses_mask'].cuda()
            combined_mask = torch.logical_and(obj_trajs_mask, obj_poses_mask)
            combined_valid_mask = (combined_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
            BN = num_center_objects * num_objects
            poses_flat = obj_poses.reshape(BN, num_timestamps, obj_poses.shape[-1])

            if self.use_cross_attn_pose:
                # H5: cross-attention — agent feature queries 11 pose tokens
                # Zero out masked frames first: NaN/garbage in masked positions × attn_weight=0 = NaN (IEEE 754)
                poses_clean = poses_flat * combined_mask.reshape(BN, num_timestamps, 1).float()
                pose_keys = self.pose_proj(poses_clean)  # (BN, T, D_MODEL)
                if self.use_pose_sinusoidal_pe:
                    pose_keys = pose_keys + self.pose_sinusoidal_pe  # broadcast over BN
                agent_query = obj_polylines_feature.reshape(BN, 1, self.model_cfg.D_MODEL)  # (BN, 1, D_MODEL)
                # key_padding_mask: True = ignore (MHA convention)
                key_padding_mask = ~combined_mask.reshape(BN, num_timestamps)
                # For all-masked agents, add a sentinel unmasked position to prevent all-inf softmax
                all_masked = key_padding_mask.all(dim=-1)  # (BN,)
                if all_masked.any():
                    key_padding_mask[all_masked, 0] = False  # unmask position 0 as dummy key
                pose_ctx, _ = self.pose_cross_attn(
                    query=agent_query, key=pose_keys, value=pose_keys,
                    key_padding_mask=key_padding_mask
                )  # (BN, 1, D_MODEL)
                # Zero out completely masked agents (sentinel gave them a garbage context)
                pose_ctx = pose_ctx * (~all_masked).view(BN, 1, 1).float()
                obj_poses_feature = pose_ctx.squeeze(1).reshape(num_center_objects, num_objects, self.model_cfg.D_MODEL)
            else:
                # Default: GRU compresses sequence to final hidden state
                obj_poses_buffer = obj_poses.new_zeros(BN, self.model_cfg.D_MODEL)
                _, final_hidden = self.pose_encoder(poses_flat)  # (num_layers, BN, D_MODEL)
                final_hidden = final_hidden[-1]  # (BN, D_MODEL)
                obj_poses_buffer[combined_valid_mask.view(-1)] = final_hidden[combined_valid_mask.view(-1)]
                obj_poses_feature = obj_poses_buffer.view(num_center_objects, num_objects, self.model_cfg.D_MODEL)

            # fuse pose feature and polyline feature
            fused_obj_feature = torch.cat((obj_polylines_feature, obj_poses_feature), dim=-1)
            if self.pose_fuser_residual:
                obj_polylines_feature = obj_polylines_feature + self.pose_fuser(fused_obj_feature)
            else:
                obj_polylines_feature = self.pose_fuser(fused_obj_feature)
        
        # apply self-attn
        obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)

        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1) 
        global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1) 
        global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1) 

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg.NUM_OF_ATTN_NEIGHBORS
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects]
        map_polylines_feature = global_token_feature[:, num_objects:]
        assert map_polylines_feature.shape[1] == num_polylines

        # organize return features
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        batch_dict['obj_mask'] = obj_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict

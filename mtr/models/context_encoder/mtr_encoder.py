# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import torch
import torch.nn as nn


from mtr.models.utils.transformer import transformer_encoder_layer, position_encoding_utils
from mtr.models.utils import polyline_encoder
from mtr.models.utils import attention_pooling
from mtr.utils import common_utils
from mtr.ops.knn import knn_utils
from mtr.models.context_encoder import pointnet
import torch.distributed as dist


class JEPAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        self.use_poses = self.model_cfg.get('USE_POSES', False)

        self.attn_pooling = self.model_cfg.get('USE_ATTN_POOL', False)

        self.lnorm = self.model_cfg.get('USE_LAYER_NORM', False)

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
        if self.attn_pooling:
            self.attention_pooling = self.build_attention_pooling(self.model_cfg.D_MODEL)
        if self.use_poses:
            self.pose_encoder = self.build_pose_encoder()
            if self.model_cfg.POSE_ENCODER.TYPE != 'PointNet':
                if self.model_cfg.POSE_ENCODER.TYPE == 'Features':
                    self.pose_sequencer = nn.GRU(64*64, self.model_cfg.POSE_ENCODER.D_MODEL_POSES, batch_first=True)
                else:
                    self.pose_sequencer = nn.GRU(self.model_cfg.POSE_ENCODER.D_MODEL_POSES, self.model_cfg.POSE_ENCODER.D_MODEL_POSES, batch_first=True)
            if self.model_cfg.FEATURE_FUSER.TYPE == 'MLP':
                self.feature_fuser = nn.Sequential(
                    nn.Linear(self.model_cfg.D_MODEL + self.model_cfg.POSE_ENCODER.D_MODEL_POSES, self.model_cfg.D_MODEL),
                    nn.ReLU()
                )
            # Value: position features, Key: position features, Query: pose features
            elif self.model_cfg.FEATURE_FUSER.TYPE == 'ATTENTION':
                self.feature_fuser = nn.MultiheadAttention(
                    embed_dim=self.model_cfg.D_MODEL,
                    num_heads=self.model_cfg.FEATURE_FUSER.NUM_HEADS_FUSER,
                    batch_first=True
                )

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


    def build_pose_encoder(self,):
        if self.model_cfg.POSE_ENCODER.TYPE == 'MLP':
            hidden_dims = self.model_cfg.POSE_ENCODER.HIDDEN_DIMS
            module_list = list()
            module_list.append(
                nn.Sequential(
                    nn.Linear(self.model_cfg.POSE_ENCODER.NUM_JOINTS*3, hidden_dims[0]),
                    nn.ReLU()
                )
            )
            for i, hidden_dim in enumerate(hidden_dims[:-1]):
                module_list.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dims[i+1]),
                    nn.ReLU()
                ))
            pose_encoder = nn.Sequential(*module_list)
        elif self.model_cfg.POSE_ENCODER.TYPE == 'PointNet':
            pose_encoder = pointnet.PointNetEncoder(hidden_dims=self.model_cfg.POSE_ENCODER.HIDDEN_DIMS, hidden_dims_conv=self.model_cfg.POSE_ENCODER.HIDDEN_DIMS_CONV, hidden_dims_fc=self.model_cfg.POSE_ENCODER.HIDDEN_DIMS_FC)
        elif self.model_cfg.POSE_ENCODER.TYPE == 'Transformer':
            self.pre_pose_encoder = nn.Sequential(
                nn.Linear(self.model_cfg.POSE_ENCODER.NUM_JOINTS*3, self.model_cfg.POSE_ENCODER.D_MODEL_POSES),
                nn.ReLU())
            pose_encoder = nn.MultiheadAttention(self.model_cfg.POSE_ENCODER.D_MODEL_POSES, self.model_cfg.POSE_ENCODER.NUM_HEADS_POSES, batch_first=True)
        
        elif self.model_cfg.POSE_ENCODER.TYPE == 'Features':
            # Use the features computed by the pose estimator
            self.pre_pose_encoder = nn.Identity()
            # pose_encoder = []
            # for _ in range(self.model_cfg.POSE_ENCODER.NUM_LAYERS_POSES):
            #     pose_encoder.append(self.build_transformer_encoder_layer(
            #         d_model=self.model_cfg.POSE_ENCODER.D_MODEL_POSES,
            #         nhead=self.model_cfg.POSE_ENCODER.NUM_HEADS_POSES,
            #         dropout=self.model_cfg.POSE_ENCODER.get('DROPOUT_POSES', 0.1),
            #         normalize_before=False,
            #         use_local_attn=self.model_cfg.get('USE_LOCAL_ATTN', False)
            #     ))
            # pose_encoder = nn.ModuleList(pose_encoder) 
        return pose_encoder


    def build_attention_pooling(self, d_model):
        return attention_pooling.AttentionPooling(feature_dim=d_model)


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
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)
 
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask_t,
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
    

    def get_jepa_loss(self, output_encoder, output_target_encoder, mse_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        num_center_objects, d_model = output_encoder.shape
        class AllReduce(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if (
                    dist.is_available()
                    and dist.is_initialized()
                    and (dist.get_world_size() > 1)
                ):
                    x = x.contiguous() / dist.get_world_size()
                    dist.all_reduce(x)
                return x

            @staticmethod
            def backward(ctx, grads):
                return grads
        # MSE loss
        mse_loss = torch.nn.functional.smooth_l1_loss(output_encoder, output_target_encoder)
        mse_loss = AllReduce.apply(mse_loss)

        # Variance loss
        # Turn encoded features into [num_center_objects, d_model]
        output_encoder = output_encoder - torch.mean(output_encoder, dim=0)
        output_target_encoder = output_target_encoder - torch.mean(output_target_encoder, dim=0)
        std_encoder = torch.sqrt(output_encoder.var(dim=0) + 0.0001)
        std_target_encoder = torch.sqrt(output_target_encoder.var(dim=0) + 0.0001)
        std_loss = torch.mean(torch.nn.functional.relu(1 - std_encoder)) / 2 + torch.mean(torch.nn.functional.relu(1 - std_target_encoder)) / 2
        std_loss = AllReduce.apply(std_loss)

        # Covariance loss
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        cov_encoder = (output_encoder.T @ output_encoder) / (num_center_objects - 1)
        cov_target_encoder = (output_target_encoder.T @ output_target_encoder) / (num_center_objects - 1)
        cov_loss = off_diagonal(cov_encoder).pow_(2).sum().div(d_model) + off_diagonal(cov_target_encoder).pow_(2).sum().div(d_model)
        cov_loss = AllReduce.apply(cov_loss)

        # Weighted loss
        loss = (mse_coeff * mse_loss + std_coeff * std_loss + cov_coeff * cov_loss)
        return loss


    def forward(self, batch_dict, target=False):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict['input_dict']
        if target == False:
            obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].cuda(), input_dict['obj_trajs_mask'].cuda()
        else:
            obj_trajs, obj_trajs_mask = input_dict['jepa_obj_trajs_future_state'].cuda(), input_dict['obj_trajs_future_mask'].cuda()
        map_polylines, map_polylines_mask = input_dict['map_polylines'].cuda(), input_dict['map_polylines_mask'].cuda()

        if self.use_poses:
            obj_poses = input_dict['pose_data'].cuda()
            obj_poses_mask = input_dict['pose_mask'].cuda()
            if self.model_cfg.POSE_ENCODER.TYPE == 'Features':
                obj_poses = input_dict['pose_features'].cuda()
            # Apply pose encoder
            num_center_objects, num_objects, num_timestamps, _, _ = obj_poses.shape
            if self.model_cfg.POSE_ENCODER.TYPE == 'PointNet':
                obj_poses = obj_poses.view(num_center_objects, num_objects, -1, 3).permute(0, 1, 3, 2).view(num_center_objects*num_objects, 3, -1)
            else:
                obj_poses = obj_poses.view(num_center_objects*num_objects, num_timestamps, -1)
            if self.model_cfg.POSE_ENCODER.TYPE == 'Transformer':
                obj_poses_feature = self.pre_pose_encoder(obj_poses)
                obj_poses_feature, _ = self.pose_encoder(obj_poses_feature, obj_poses_feature, obj_poses_feature)
            else:
                obj_poses_feature = self.pose_encoder(obj_poses)
            if self.model_cfg.POSE_ENCODER.TYPE == 'PointNet':
                obj_poses_feature = obj_poses_feature[0].view(num_center_objects, num_objects, -1)
            else:
                if self.model_cfg.POSE_ENCODER.TYPE != 'Features':
                    obj_poses_feature = obj_poses_feature.view(-1, num_timestamps, self.model_cfg.POSE_ENCODER.D_MODEL_POSES)
                _, obj_poses_feature = self.pose_sequencer(obj_poses_feature)
                obj_poses_feature = obj_poses_feature.permute(1, 0, 2).view(num_center_objects, num_objects, -1)



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

        # fuse pose features with object features
        if self.use_poses:
            if self.model_cfg.FEATURE_FUSER.TYPE == 'MLP':
                # obj_polylines_feature: shape [38, 122, 256]
                # obj_poses_feature: shape [38, 122, 11, 256]
                obj_polylines_feature = self.feature_fuser(torch.cat((obj_polylines_feature, obj_poses_feature), dim=-1))
            elif self.model_cfg.FEATURE_FUSER.TYPE == 'ATTENTION':
                obj_polylines_feature, _ = self.feature_fuser(value=obj_polylines_feature, key=obj_polylines_feature, query=obj_poses_feature)


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

        if self.lnorm and target == False:
            center_objects_feature = torch.nn.functional.layer_norm(center_objects_feature, center_objects_feature.shape[1:])

        if target == False:

            batch_dict['center_objects_feature'] = center_objects_feature
            batch_dict['obj_feature'] = obj_polylines_feature
            batch_dict['map_feature'] = map_polylines_feature
            batch_dict['obj_mask'] = obj_valid_mask
            batch_dict['map_mask'] = map_valid_mask
            batch_dict['obj_pos'] = obj_trajs_last_pos
            batch_dict['map_pos'] = map_polylines_center

            if self.attn_pooling:
                if not self.lnorm:
                    batch_dict['pooled_attn'] = self.attention_pooling(obj_polylines_feature)
                else:
                    batch_dict['pooled_attn'] = torch.nn.functional.layer_norm(self.attention_pooling(obj_polylines_feature), (obj_polylines_feature.shape[0], obj_polylines_feature.shape[-1]))
            return batch_dict
        else:
            if self.attn_pooling:
                return self.attention_pooling(obj_polylines_feature)
            else:
                return center_objects_feature



class MTREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        self.use_poses = self.model_cfg.get('USE_POSES', False)

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
        if self.use_poses:
            self.pose_encoder = self.build_pose_encoder()
            if self.model_cfg.POSE_ENCODER.TYPE != 'PointNet':
                if self.model_cfg.POSE_ENCODER.TYPE == 'Features':
                    self.pose_sequencer = nn.GRU(64*64, self.model_cfg.POSE_ENCODER.D_MODEL_POSES, batch_first=True)
                else:
                    self.pose_sequencer = nn.GRU(self.model_cfg.POSE_ENCODER.D_MODEL_POSES, self.model_cfg.POSE_ENCODER.D_MODEL_POSES, batch_first=True)
            if self.model_cfg.FEATURE_FUSER.TYPE == 'MLP':
                self.feature_fuser = nn.Sequential(
                    nn.Linear(self.model_cfg.D_MODEL + self.model_cfg.POSE_ENCODER.D_MODEL_POSES, self.model_cfg.D_MODEL),
                    nn.ReLU()
                )
            # Value: position features, Key: position features, Query: pose features
            elif self.model_cfg.FEATURE_FUSER.TYPE == 'ATTENTION':
                self.feature_fuser = nn.MultiheadAttention(
                    embed_dim=self.model_cfg.D_MODEL,
                    num_heads=self.model_cfg.FEATURE_FUSER.NUM_HEADS_FUSER,
                    batch_first=True
                )

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


    def build_pose_encoder(self,):
        if self.model_cfg.POSE_ENCODER.TYPE == 'MLP':
            hidden_dims = self.model_cfg.POSE_ENCODER.HIDDEN_DIMS
            module_list = list()
            module_list.append(
                nn.Sequential(
                    nn.Linear(self.model_cfg.POSE_ENCODER.NUM_JOINTS*3, hidden_dims[0]),
                    nn.ReLU()
                )
            )
            for i, hidden_dim in enumerate(hidden_dims[:-1]):
                module_list.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dims[i+1]),
                    nn.ReLU()
                ))
            pose_encoder = nn.Sequential(*module_list)
        elif self.model_cfg.POSE_ENCODER.TYPE == 'PointNet':
            pose_encoder = pointnet.PointNetEncoder(hidden_dims=self.model_cfg.POSE_ENCODER.HIDDEN_DIMS, hidden_dims_conv=self.model_cfg.POSE_ENCODER.HIDDEN_DIMS_CONV, hidden_dims_fc=self.model_cfg.POSE_ENCODER.HIDDEN_DIMS_FC)
        elif self.model_cfg.POSE_ENCODER.TYPE == 'Transformer':
            self.pre_pose_encoder = nn.Sequential(
                nn.Linear(self.model_cfg.POSE_ENCODER.NUM_JOINTS*3, self.model_cfg.POSE_ENCODER.D_MODEL_POSES),
                nn.ReLU())
            pose_encoder = nn.MultiheadAttention(self.model_cfg.POSE_ENCODER.D_MODEL_POSES, self.model_cfg.POSE_ENCODER.NUM_HEADS_POSES, batch_first=True)
        
        elif self.model_cfg.POSE_ENCODER.TYPE == 'Features':
            # Use the features computed by the pose estimator
            self.pre_pose_encoder = nn.Identity()
            # pose_encoder = []
            # for _ in range(self.model_cfg.POSE_ENCODER.NUM_LAYERS_POSES):
            #     pose_encoder.append(self.build_transformer_encoder_layer(
            #         d_model=self.model_cfg.POSE_ENCODER.D_MODEL_POSES,
            #         nhead=self.model_cfg.POSE_ENCODER.NUM_HEADS_POSES,
            #         dropout=self.model_cfg.POSE_ENCODER.get('DROPOUT_POSES', 0.1),
            #         normalize_before=False,
            #         use_local_attn=self.model_cfg.get('USE_LOCAL_ATTN', False)
            #     ))
            # pose_encoder = nn.ModuleList(pose_encoder) 
        return pose_encoder


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
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)
 
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask_t,
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

        if self.use_poses:
            obj_poses = input_dict['pose_data'].cuda()
            obj_poses_mask = input_dict['pose_mask'].cuda()
            if self.model_cfg.POSE_ENCODER.TYPE == 'Features':
                obj_poses = input_dict['pose_features'].cuda()
            # Apply pose encoder
            num_center_objects, num_objects, num_timestamps, _, _ = obj_poses.shape
            if self.model_cfg.POSE_ENCODER.TYPE == 'PointNet':
                obj_poses = obj_poses.view(num_center_objects, num_objects, -1, 3).permute(0, 1, 3, 2).view(num_center_objects*num_objects, 3, -1)
            else:
                obj_poses = obj_poses.view(num_center_objects*num_objects, num_timestamps, -1)
            if self.model_cfg.POSE_ENCODER.TYPE == 'Transformer':
                obj_poses_feature = self.pre_pose_encoder(obj_poses)
                obj_poses_feature, _ = self.pose_encoder(obj_poses_feature, obj_poses_feature, obj_poses_feature)
            else:
                obj_poses_feature = self.pose_encoder(obj_poses)
            if self.model_cfg.POSE_ENCODER.TYPE == 'PointNet':
                obj_poses_feature = obj_poses_feature[0].view(num_center_objects, num_objects, -1)
            else:
                if self.model_cfg.POSE_ENCODER.TYPE != 'Features':
                    obj_poses_feature = obj_poses_feature.view(-1, num_timestamps, self.model_cfg.POSE_ENCODER.D_MODEL_POSES)
                _, obj_poses_feature = self.pose_sequencer(obj_poses_feature)
                obj_poses_feature = obj_poses_feature.permute(1, 0, 2).view(num_center_objects, num_objects, -1)



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

        # fuse pose features with object features
        if self.use_poses:
            if self.model_cfg.FEATURE_FUSER.TYPE == 'MLP':
                # obj_polylines_feature: shape [38, 122, 256]
                # obj_poses_feature: shape [38, 122, 11, 256]
                obj_polylines_feature = self.feature_fuser(torch.cat((obj_polylines_feature, obj_poses_feature), dim=-1))
            elif self.model_cfg.FEATURE_FUSER.TYPE == 'ATTENTION':
                obj_polylines_feature, _ = self.feature_fuser(value=obj_polylines_feature, key=obj_polylines_feature, query=obj_poses_feature)


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

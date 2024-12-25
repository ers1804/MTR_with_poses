# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from .jepa_utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from .masks.utils import apply_masks

from mtr.models.utils.attention_pooling import AttentionPoolingTimesteps, AttentionPooling


def get_time_sincos_pos_embed(embed_dim, len, cls_token=False):
    """
    len: int of the sequence length
    return:
    time_embed: [len, embed_dim] or [1+len, embed_dim] (w/ or w/o cls_token)
    """
    time = np.arange(len, dtype=float)
    time_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, time)
    if cls_token:
        time_embed = np.concatenate([np.zeros([1, embed_dim]), time_embed], axis=0)
    return time_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn[torch.all(attn == float('-inf'), dim=-1)] = 0.
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
    

class AttentionWithMask(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape  # B = batch size, N = sequence length, C = feature dimension
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)

        if mask is not None:
            mask = mask.unsqueeze(1)  # Broadcast mask: (B, 1, N, N)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn[torch.all(attn == float('-inf'), dim=-1)] = 0.
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
    

class AgentTimeAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.agent_attention = AttentionWithMask(dim, num_heads)
        self.time_attention = AttentionWithMask(dim, num_heads)

    def forward(self, x, valid_mask):
        """
        Args:
            x: Tensor of shape (num_center_objects, num_objects, num_timesteps, num_attributes)
            valid_mask: Boolean mask of shape (num_center_objects, num_objects, num_timesteps)
                        where True indicates valid timesteps and False indicates invalid ones.
        Returns:
            Updated tensor with agent-wise and time-wise attention applied, maintaining the same shape.
        """
        num_center_objects, num_objects, num_timesteps, num_attributes = x.shape

        # --- Process each agent-centric scene separately ---
        results = []
        for i in range(num_center_objects):
            # Isolate the i-th scene
            x_scene = x[i]  # Shape: (num_objects, num_timesteps, num_attributes)
            valid_scene_mask = valid_mask[i]  # Shape: (num_objects, num_timesteps)

            # --- Agent-Wise Attention ---
            # Transpose to focus on agents: (num_timesteps, num_objects, num_attributes)
            x_agent = x_scene.permute(1, 0, 2)  # Shape: (num_timesteps, num_objects, num_attributes)

            # Mask preparation for agent-wise attention
            agent_mask = valid_scene_mask.permute(1, 0)  # Shape: (num_timesteps, num_objects)
            agent_mask = agent_mask.unsqueeze(1) * agent_mask.unsqueeze(2)  # Shape: (num_timesteps, num_objects, num_objects)

            # Apply agent-wise attention
            x_agent = x_agent.reshape(-1, num_objects, num_attributes)  # Shape: (num_timesteps, num_objects, num_attributes)
            agent_mask = agent_mask.reshape(-1, num_objects, num_objects)  # Flatten to batch size
            x_agent, _ = self.agent_attention(x_agent, mask=agent_mask)
            x_agent = x_agent.reshape(num_timesteps, num_objects, -1)  # Restore shape

            # Transpose back: (num_objects, num_timesteps, num_attributes)
            x_scene = x_agent.permute(1, 0, 2)

            # --- Time-Wise Attention ---
            # Mask preparation for time-wise attention
            time_mask = valid_scene_mask.unsqueeze(1) * valid_scene_mask.unsqueeze(2)  # Shape: (num_objects, num_timesteps, num_timesteps)

            # Apply time-wise attention
            x_time = x_scene.reshape(-1, num_timesteps, num_attributes)  # Shape: (num_objects, num_timesteps, num_attributes)
            time_mask = time_mask.reshape(-1, num_timesteps, num_timesteps)  # Flatten to batch size
            x_time, _ = self.time_attention(x_time, mask=time_mask)
            x_time = x_time.reshape(num_objects, num_timesteps, -1)  # Restore shape

            # Collect the processed scene
            results.append(x_time)

        # Combine all processed scenes back into a single tensor
        x = torch.stack(results, dim=0)  # Shape: (num_center_objects, num_objects, num_timesteps, num_attributes)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TimeStepEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_features_dim=110, hidden_features=256, out_features=512, act_layer=nn.GELU, drop=0., layer_norm=False):
        super().__init__()

        self.proj = MLP(in_features=in_features_dim, hidden_features=hidden_features, out_features=out_features, act_layer=act_layer, drop=drop)
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None

    def forward(self, x):
        """
        x: num_center_objects, num_objects, num_timesteps, in_features_dim
        return: num_center_objects, num_objects, num_timesteps, out_features
        """
        #B, C, H, W = x.shape
        x = self.proj(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class TrajectoryTransformerPredictor(nn.Module):
    """ Trajectory Transformer """
    def __init__(
        self,
        embed_dim=256,
        predictor_embed_dim=128,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        # self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
        #                                         requires_grad=False)
        # predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
        #                                               int(num_patches**.5),
        #                                               cls_token=False)
        # self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        # if not isinstance(masks_x, list):
        #     masks_x = [masks_x]

        # if not isinstance(masks, list):
        #     masks = [masks]

        # -- Batch Size
        #B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(torch.unsqueeze(x, dim=1))

        # -- add positional embedding to x tokens
        # x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        # x += apply_masks(x_pos_embed, masks_x)

        #_, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        # pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        # pos_embs = apply_masks(pos_embs, masks)
        # pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # # --
        # pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # # --
        # pred_tokens += pos_embs
        # x = x.repeat(len(masks), 1, 1)
        # x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        #x = x[:, N_ctxt:]
        x = self.predictor_proj(x)
        return torch.squeeze(x, dim=1)


class TrajectoryTransformer(nn.Module):
    """ Trajectory Transformer """
    def __init__(
        self,
        in_features=110,
        embed_hidden_dim=256,
        embed_act_layer=nn.GELU,
        embed_dropout=0.,
        embed_layer_norm=False,
        num_total_timesteps=91,
        pre_num_heads=4,
        embed_dim=256,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.timestep_embed = TimeStepEmbed(
            in_features_dim=in_features,
            hidden_features=embed_hidden_dim,
            out_features=embed_dim,
            act_layer=embed_act_layer,
            drop=embed_dropout,
            layer_norm=embed_layer_norm)
        self.num_total_timesteps=num_total_timesteps
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_total_timesteps, embed_dim), requires_grad=False)
        pos_embed = get_time_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(num_total_timesteps),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0).unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()
        # ------
        self.agent_time_attention = AgentTimeAttention(embed_dim, pre_num_heads)
        self.attention_pooling = AttentionPoolingTimesteps(embed_dim)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, agent_valid_mask=None, target=False, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- embed timesteps x
        x = self.timestep_embed(x)
        num_center_objects, num_objects, num_timesteps, embed_features = x.shape

        # -- add positional embedding to x
        #pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        if not target:
            x = x + self.pos_embed[:, :, :num_timesteps, :]
        else:
            total_timesteps = self.pos_embed.shape[2]
            x = x + self.pos_embed[:, :, (total_timesteps - num_timesteps):, :]
        # -- apply agent-time attention
        x = self.agent_time_attention(x, agent_valid_mask)
        # -- pool features across time dimension
        x = self.attention_pooling(x, mask=agent_valid_mask)

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[2] - 1
        N = pos_embed.shape[2] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = TrajectoryTransformerPredictor(
        in_features=110,
        embed_hidden_dim=256,
        embed_act_layer=nn.GELU,
        embed_dropout=0.,
        embed_layer_norm=False,
        num_total_timesteps=91,
        pre_num_heads=4,
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(**kwargs):
    model = TrajectoryTransformer(
        in_features=110,
        embed_hidden_dim=256,
        embed_act_layer=nn.GELU,
        embed_dropout=0.,
        embed_layer_norm=True,
        num_total_timesteps=91,
        pre_num_heads=4,
        embed_dim=256, depth=12, num_heads=4, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(**kwargs):
    model = TrajectoryTransformer(
        in_features=110,
        embed_hidden_dim=256,
        embed_act_layer=nn.GELU,
        embed_dropout=0.,
        embed_layer_norm=False,
        num_total_timesteps=91,
        pre_num_heads=4,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(**kwargs):
    model = TrajectoryTransformer(
        in_features=110,
        embed_hidden_dim=256,
        embed_act_layer=nn.GELU,
        embed_dropout=0.,
        embed_layer_norm=False,
        num_total_timesteps=91,
        pre_num_heads=4,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(**kwargs):
    model = TrajectoryTransformer(
        in_features=110,
        embed_hidden_dim=256,
        embed_act_layer=nn.GELU,
        embed_dropout=0.,
        embed_layer_norm=False,
        num_total_timesteps=91,
        pre_num_heads=4,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(**kwargs):
    model = TrajectoryTransformer(
        in_features=110,
        embed_hidden_dim=256,
        embed_act_layer=nn.GELU,
        embed_dropout=0.,
        embed_layer_norm=False,
        num_total_timesteps=91,
        pre_num_heads=4,
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(**kwargs):
    model = TrajectoryTransformer(
        in_features=110,
        embed_hidden_dim=256,
        embed_act_layer=nn.GELU,
        embed_dropout=0.,
        embed_layer_norm=False,
        num_total_timesteps=91,
        pre_num_heads=4,
        embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}

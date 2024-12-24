import torch
import torch.nn as nn
from ..utils import common_layers
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionPooling, self).__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, encoded_scene, mask=None):
        # encoded_scene: [num_agents_of_interest, num_agents, feature_dim]
        _, num_agents, _ = encoded_scene.shape
        queries = self.query(encoded_scene)  # [num_agents_of_interest, num_agents, feature_dim]
        keys = self.key(encoded_scene)      # [num_agents_of_interest, num_agents, feature_dim]
        values = self.value(encoded_scene)  # [num_agents_of_interest, num_agents, feature_dim]
        
        # Compute attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(self.feature_dim, dtype=torch.float32))  # [num_agents_of_interest, num_agents, num_agents]

        if mask is not None:

            counting_mask = mask

            mask = mask.unsqueeze(1).expand(-1, num_agents, -1)  # [num_agents_of_interest, num_agents, num_agents]

            scores = scores.masked_fill(mask == 0, float('-inf'))

            weights = self.softmax(scores)  # [num_agents_of_interest, num_agents, num_agents]

            weights = weights * mask
        else:
            weights = self.softmax(scores)
        
        # Compute weighted sum of values
        attended_representation = torch.bmm(weights, values)  # [num_agents_of_interest, num_agents, feature_dim]

        if mask is not None:
            # Normalize the agent representation by dividing through the number of valid agents
            valid_agents_count = counting_mask.sum(dim=1, keepdim=True).float()  # [num_agents_of_interest, 1]
            valid_agents_count = valid_agents_count.expand(-1, self.feature_dim)  # [num_agents_of_interest, feature_dim]
            
            # Aggregate information across agents
            return torch.sum(attended_representation, dim=1) / (valid_agents_count + 1e-9)  # [num_agents_of_interest, feature_dim]
        else:
            return torch.sum(attended_representation, dim=1) / (num_agents + 1e-9)
        

class AttentionPoolingTimesteps(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionPoolingTimesteps, self).__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, encoded_scene, mask=None):
        # encoded_scene: [B, N, T, C]
        B, N, T, C = encoded_scene.shape
        queries = self.query(encoded_scene)  # [B, N, T, C]
        keys = self.key(encoded_scene)      # [B, N, T, C]
        values = self.value(encoded_scene)  # [B, N, T, C]
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_dim, dtype=torch.float32))  # [B, N, T, T]

        if mask is not None:
            # Prepare the mask
            ori_agent_mask = mask
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, T)  # [B, N, T, T]

            # Apply the mask to the attention scores
            scores = scores.masked_fill(~mask, float('-inf'))
            print(torch.any(torch.isnan(scores)))
            # Compute attention weights
            weights = self.softmax(scores)  # [B, N, T, T]
            print(torch.any(torch.isnan(weights)))
            # Zero out invalid weights
            weights[~mask] = 0.
        else:
            weights = self.softmax(scores)
        print(torch.any(torch.isnan(weights)))
        
        # Compute weighted sum of values
        attended_representation = torch.einsum("bntt, bntc->bntc", weights, values)  # [B, N, T, C]
        print(torch.any(torch.isnan(attended_representation)))

        if mask is not None:
            # Normalize by the number of valid timesteps
            valid_timesteps_count = ori_agent_mask.sum(dim=2, keepdim=True).float()  # [B, N, 1]
            valid_timesteps_count = valid_timesteps_count.expand(-1, -1, C)  # [B, N, C]

            # Compute the final representation
            attended_representation = attended_representation.sum(dim=2) / (valid_timesteps_count + 1e-9)  # [B, N, C]
            attended_representation[valid_timesteps_count == 0] = 0.
            return attended_representation
        else:
            return attended_representation.sum(dim=2) / (T + 1e-1)  # [B, N, C]
        

# class AttentionPoolingTimesteps(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.attention_weights = nn.Linear(input_dim, 1)  # Learnable attention weights

#     def forward(self, x, valid_mask):
#         """
#         Args:
#             x: Tensor of shape (num_center_objects, num_objects, num_timesteps, num_attributes)
#             valid_mask: Boolean mask of shape (num_center_objects, num_objects, num_timesteps)
#                         where True indicates valid timesteps and False indicates invalid ones.

#         Returns:
#             Tensor of shape (num_center_objects, num_objects, num_attributes) with pooled features.
#         """
#         # Input shape: (num_center_objects, num_objects, num_timesteps, num_attributes)
#         scores = self.attention_weights(x).squeeze(-1)  # Shape: (num_center_objects, num_objects, num_timesteps)

#         # Mask out invalid timesteps by setting scores to a large negative value (-inf)
#         scores = scores.masked_fill(~valid_mask, float('-inf'))  # Shape: (num_center_objects, num_objects, num_timesteps)

#         # Compute attention weights
#         attn_weights = F.softmax(scores, dim=-1)  # Shape: (num_center_objects, num_objects, num_timesteps)

#         # Apply the mask to the attention weights to ensure invalid timesteps contribute zero
#         attn_weights = attn_weights * valid_mask  # Shape: (num_center_objects, num_objects, num_timesteps)

#         # Normalize the attention weights to ensure they sum to 1 across valid timesteps
#         attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)  # Avoid division by zero

#         # Perform weighted sum over timesteps
#         pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=2)  # Shape: (num_center_objects, num_objects, num_attributes)

#         return pooled
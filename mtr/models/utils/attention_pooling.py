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
        queries = self.query(encoded_scene)  # [num_agents_of_interest, num_agents, feature_dim]
        keys = self.key(encoded_scene)      # [num_agents_of_interest, num_agents, feature_dim]
        values = self.value(encoded_scene)  # [num_agents_of_interest, num_agents, feature_dim]
        
        # Compute attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(self.feature_dim, dtype=torch.float32))  # [num_agents_of_interest, num_agents, num_agents]

        if mask is not None:

            mask = mask.unsqueeze(1).expand(-1, scores.size(1), -1)  # [num_agents_of_interest, num_agents, num_agents]

            scores = scores.masked_fill(mask == 0, float('-inf'))

            weights = self.softmax(scores)  # [num_agents_of_interest, num_agents, num_agents]

            weights = weights * mask
        else:
            weights = self.softmax(scores)
        
        # Compute weighted sum of values
        attended_representation = torch.bmm(weights, values)  # [num_agents_of_interest, num_agents, feature_dim]
        
        # Aggregate information across agents
        return torch.sum(attended_representation, dim=1)  # [num_agents_of_interest, feature_dim]
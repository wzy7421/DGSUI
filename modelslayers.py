import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContinuousTimeAwareMHSA(nn.Module):
    """
    Dynamic Knowledge Representation via Continuous Temporal Awareness
    Implements Eq (1) & (2) from the paper.
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # Learnable decay rate alpha for the Ebbinghaus forgetting curve
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, time_intervals, mask=None):
        # x: [batch_size, seq_len, hidden_size]
        # time_intervals: [batch_size, seq_len, seq_len] representing \Delta t

        batch_size, seq_len, _ = x.size()

        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Eq (1): Physical time-decay function f(\Delta t) = exp(-\alpha * \Delta t)
        # Eq (2): Bias injection -> log f(\Delta t) = -\alpha * \Delta t
        time_decay_bias = - torch.abs(self.alpha) * time_intervals
        time_decay_bias = time_decay_bias.unsqueeze(1)  # Broadcast over heads

        scores = scores + time_decay_bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return out


class OrthogonalDisentanglementEngine(nn.Module):
    """
    Intent Disentanglement Engine via Orthogonal Contrastive Learning
    Implements Eq (3), (4) & (5).
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp_global = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mlp_salient = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, h_dyn):
        # h_dyn: Fused dynamic knowledge representation
        g = self.mlp_global(h_dyn)  # Steady-state Habitual Intent
        s = self.mlp_salient(h_dyn)  # Transient Exploratory Intent
        return g, s


class IntentAwareGraphRouting(nn.Module):
    """
    Intent-aware multi-layer graph routing and inference.
    Executes Eq (6) & (7).
    """

    def __init__(self):
        super().__init__()

    def forward(self, intent_emb, neighbor_emb):
        # intent_emb: [batch, hidden_size] (either g or s)
        # neighbor_emb: [batch, num_neighbors, hidden_size]

        # Calculate routing weights
        scores = torch.bmm(neighbor_emb, intent_emb.unsqueeze(-1)).squeeze(-1)
        routing_weights = F.softmax(scores, dim=-1)

        # Aggregation
        aggregated_features = torch.bmm(routing_weights.unsqueeze(1), neighbor_emb).squeeze(1)
        return aggregated_features, routing_weights
import torch
import torch.nn as nn
from .layers import ContinuousTimeAwareMHSA, OrthogonalDisentanglementEngine, IntentAwareGraphRouting


class DGSUI(nn.Module):
    def __init__(self, num_users, num_items, hidden_size=64, num_heads=4, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Base embeddings
        self.user_embedding = nn.Embedding(num_users, hidden_size)
        self.item_embedding = nn.Embedding(num_items, hidden_size)

        # Modules
        self.temporal_mhsa = ContinuousTimeAwareMHSA(hidden_size, num_heads)
        self.disentangler = OrthogonalDisentanglementEngine(hidden_size)
        self.graph_router = IntentAwareGraphRouting()

    def forward(self, user_ids, seq_item_ids, time_intervals, adj_matrix=None):
        """
        Forward pass mirroring Fig. 9 in the paper.
        """
        # 1. Look up sequences
        seq_emb = self.item_embedding(seq_item_ids)

        # 2. Dynamic Knowledge Representation (Continuous Temporal Awareness)
        h_dyn = self.temporal_mhsa(seq_emb, time_intervals)

        # Use the last hidden state representing current dynamic preference
        h_current = h_dyn[:, -1, :]

        # 3. Intent Disentanglement (Orthogonal Engine)
        g, s = self.disentangler(h_current)

        # --- Graph Routing Logic (Placeholder for high-order propagation) ---
        # In a full implementation, you would fetch multi-hop neighbors here using PyG
        # Here we simulate fetching 1-hop item neighbors for demonstration
        # neighbor_emb = fetch_neighbors(adj_matrix, user_ids)

        # Simulate dummy neighbor embeddings for the forward pass template
        batch_size = user_ids.size(0)
        dummy_neighbors = torch.randn(batch_size, 100, self.hidden_size).to(h_current.device)

        # 4. Intent-Aware Routing
        g_aggregated, _ = self.graph_router(g, dummy_neighbors)
        s_aggregated, routing_weights_s = self.graph_router(s, dummy_neighbors)

        # Final User Representation is the fusion of habit and exploration
        final_user_emb = self.user_embedding(user_ids) + g_aggregated + s_aggregated

        return final_user_emb, h_current, g, s, routing_weights_s

    def predict(self, user_emb, item_ids):
        item_emb = self.item_embedding(item_ids)
        scores = torch.sum(user_emb * item_emb, dim=-1)
        return scores
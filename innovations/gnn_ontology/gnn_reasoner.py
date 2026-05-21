"""
GNN Reasoner: Graph Attention Network (GAT) for clinical knowledge reasoning
over the chest X-ray knowledge graph.

Uses multi-head graph attention to propagate information between disease nodes,
enabling:
    - Inference of unobserved findings from observed ones
    - Knowledge-based feature refinement
    - Cross-disease attention for contextualized representations

Architecture:
    GATLayer (×2) -> Node Features -> Readout -> Knowledge Vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GATLayer(nn.Module):
    """Single Graph Attention Network layer.

    For each edge (i->j), computes attention coefficient alpha_ij indicating
    the importance of node i's features to node j.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        concat: Whether to concatenate multi-head outputs (True) or average (False)
        negative_slope: LeakyReLU negative slope
    """

    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1,
                 concat=True, negative_slope=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope

        # Linear projection for each head
        self.W = nn.Parameter(torch.zeros(num_heads, in_features, out_features))
        nn.init.xavier_uniform_(self.W)

        # Attention parameters
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_features, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if not concat:
            self.output_proj = nn.Linear(out_features * num_heads, out_features)

    def forward(self, x, edge_index):
        """GAT forward pass.

        Args:
            x: (N, in_features) node features
            edge_index: (2, E) edge indices

        Returns:
            out: (N, out_features * num_heads) if concat, else (N, out_features)
        """
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Linear projection per head: (N, H, out_features)
        x_proj = torch.einsum('ni,hio->nho', x, self.W)

        # Attention scores
        attn_src = (x_proj[src] * self.a_src.unsqueeze(0)).sum(dim=-1)  # (E, H)
        attn_dst = (x_proj[dst] * self.a_dst.unsqueeze(0)).sum(dim=-1)  # (E, H)
        attn_raw = attn_src + attn_dst  # (E, H)
        attn_raw = self.leaky_relu(attn_raw)

        # Softmax per target node
        attn_weights = torch.zeros(N, self.num_heads, device=x.device)
        attn_exp = torch.exp(attn_raw)
        attn_weights = attn_weights.index_add(0, dst, attn_exp)
        attn_weights = attn_weights.clamp(min=1e-8)
        attn_norm = attn_exp / attn_weights[dst]

        # Weighted message aggregation
        attn_norm = self.dropout(attn_norm)
        messages = x_proj[src] * attn_norm.unsqueeze(-1)  # (E, H, out)

        out = torch.zeros(N, self.num_heads, self.out_features, device=x.device)
        out = out.index_add(0, dst, messages)

        if self.concat:
            out = out.reshape(N, -1)  # (N, H * out_features)
        else:
            out = out.mean(dim=1)  # (N, out_features)

        return out


class GNNReasoner(nn.Module):
    """Two-layer GAT for clinical knowledge reasoning.

    Args:
        d_model: Feature dimension
        num_labels: Number of CheXpert labels
        num_heads: GAT attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model=512, num_labels=14, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels

        # Graph attention layers
        gat_hidden = d_model // num_heads
        self.gat1 = GATLayer(d_model, gat_hidden, num_heads, dropout, concat=True)
        self.gat2 = GATLayer(gat_hidden * num_heads, d_model, 1, dropout, concat=False)

        # Residual projection
        self.residual_proj = nn.Linear(d_model, d_model)

        # Knowledge readout: pool node features to global knowledge
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Cross-disease attention
        self.disease_cross_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True,
        )

    def forward(self, node_features, edge_index):
        """Run GNN reasoning over the knowledge graph.

        Args:
            node_features: (num_labels, d_model) input node features
            edge_index: (2, num_edges) graph edges

        Returns:
            reasoned_features: (num_labels, d_model) knowledge-enhanced features
            global_knowledge: (d_model,) global graph-level representation
        """
        # First GAT layer
        h1 = self.gat1(node_features, edge_index)
        h1 = F.elu(h1)

        # Second GAT layer
        h2 = self.gat2(h1, edge_index)

        # Residual connection
        reasoned = h2 + self.residual_proj(node_features)
        reasoned = F.layer_norm(reasoned, (self.d_model,))

        # Per-node readout
        reasoned = self.readout(reasoned)

        # Cross-disease attention for global reasoning
        reasoned_unsq = reasoned.unsqueeze(0)  # (1, num_labels, d_model)
        attended, _ = self.disease_cross_attn(reasoned_unsq, reasoned_unsq, reasoned_unsq)
        reasoned = attended.squeeze(0)

        # Global knowledge vector (attention-weighted pooling)
        attn_weights = F.softmax(
            torch.matmul(reasoned, reasoned.mean(dim=0, keepdim=True).T), dim=0
        )
        global_knowledge = (attn_weights * reasoned).sum(dim=0)

        return reasoned, global_knowledge

    def predict_comorbidities(self, node_features):
        """Predict comorbidity links between disease nodes using dot-product attention.

        Args:
            node_features: (num_labels, d_model) reasoned features

        Returns:
            link_scores: (num_labels, num_labels) predicted link probabilities
        """
        normed = F.normalize(node_features, dim=-1)
        link_scores = torch.matmul(normed, normed.T)
        return torch.sigmoid(link_scores)

"""
Gated Fusion: Learnable gating mechanism that fuses GNN-reasoned knowledge
with the original attention-based memory retrieval from LAMRG.

Fusion strategy:
    gate = σ(W_g [mem_prior; gnn_knowledge])
    fused = gate * mem_prior + (1 - gate) * gnn_knowledge

This enables the model to learn WHEN to rely on structured medical knowledge
(via GNN) vs. data-driven learned patterns (via memory attention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """Gated fusion between memory attention prior and GNN-reasoned knowledge.

    Args:
        d_model: Feature dimension
        num_labels: Number of findings
        use_context_gate: Whether gate is context-dependent
        use_label_gate: Whether to compute per-label gating
    """

    def __init__(self, d_model=512, num_labels=14, use_context_gate=True,
                 use_label_gate=True):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.use_context_gate = use_context_gate
        self.use_label_gate = use_label_gate

        # Global gate: sample-level fusion ratio
        self.global_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Per-label gate: fine-grained fusion per disease
        if use_label_gate:
            self.label_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        # Context gate: image features modulate the fusion
        if use_context_gate:
            self.context_gate = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        # Post-fusion refinement
        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, memory_prior, gnn_knowledge, context_features=None,
                label_predictions=None):
        """Fuse memory retrieval with GNN-reasoned knowledge.

        Args:
            memory_prior: (B, num_labels, d_model) from attention-based memory
            gnn_knowledge: (num_labels, d_model) or (B, num_labels, d_model) from GNN
            context_features: Optional (B, d_model) image context
            label_predictions: Optional (B, num_labels) predicted finding probabilities

        Returns:
            fused_knowledge: (B, num_labels, d_model) gated fusion
            gate_values: Dict with gate activation statistics
        """
        bsz = memory_prior.shape[0]

        # Expand GNN knowledge if needed
        if gnn_knowledge.dim() == 2:
            gnn_knowledge = gnn_knowledge.unsqueeze(0).expand(bsz, -1, -1)

        # Global gate
        mem_pooled = memory_prior.mean(dim=1)  # (B, d_model)
        gnn_pooled = gnn_knowledge.mean(dim=1)  # (B, d_model)
        global_gate_input = torch.cat([mem_pooled, gnn_pooled], dim=-1)
        global_gate_val = self.global_gate(global_gate_input)  # (B, 1)

        # Per-label gate
        if self.use_label_gate:
            label_pair = torch.cat([memory_prior, gnn_knowledge], dim=-1)
            label_gate_val = self.label_gate(label_pair)  # (B, num_labels, 1)
        else:
            label_gate_val = global_gate_val.unsqueeze(1).expand(-1, self.num_labels, -1)

        # Context gate
        if self.use_context_gate and context_features is not None:
            ctx_expanded = context_features.unsqueeze(1).expand(-1, self.num_labels, -1)
            context_input = torch.cat([memory_prior, gnn_knowledge, ctx_expanded], dim=-1)
            context_gate_val = self.context_gate(context_input)  # (B, num_labels, 1)
            gate = context_gate_val * label_gate_val
        else:
            gate = label_gate_val

        # Fuse
        fused = gate * memory_prior + (1.0 - gate) * gnn_knowledge

        # Refine
        fused = self.refine(fused)

        gate_stats = {
            'global_gate': global_gate_val.mean().item(),
            'label_gate_mean': label_gate_val.mean().item(),
            'label_gate_std': label_gate_val.std().item(),
        }

        return fused, gate_stats

    def compute_fusion_loss(self, gate_values):
        """Encourage the gate to make decisive choices (not always 0.5).

        Penalizes gate values near 0.5 to avoid lazy averaging.

        Args:
            gate_values: (B, num_labels, 1) learned gate values

        Returns:
            decision_loss: Scalar loss encouraging decisive gating
        """
        # Entropy minimization: push gate toward 0 or 1
        gate = gate_values.squeeze(-1)
        entropy = -(gate * torch.log(gate + 1e-8) +
                     (1 - gate) * torch.log(1 - gate + 1e-8))
        return entropy.mean()

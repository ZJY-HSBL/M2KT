"""
Memory Update Gate: Learnable gating mechanism for controlled knowledge evolution.

Implements a write gate (how much new info to incorporate) and a forget gate
(how much old info to retain) inspired by the LSTM/GRU paradigm, adapted for
the multi-head attention-based memory update in the LAMRG framework.

Key insight: Without gating, every training step overwrites memory
indiscriminately. The gate enables:
    - Retaining stable disease patterns across similar cases
    - Forgetting noise from rare or anomalous reports
    - Adaptive learning rates per memory slot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryUpdateGate(nn.Module):
    """Gated memory update mechanism for controlled knowledge accumulation.

    Args:
        d_model: Feature dimension
        bias: Learnable bias for gate initialization (negative = conservative)
    """

    def __init__(self, d_model=512, bias=-2.0):
        super().__init__()
        self.d_model = d_model

        # Write gate: how much of the new memory to accept
        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Sigmoid(),
        )

        # Forget gate: how much of the old memory to retain
        self.forget_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Sigmoid(),
        )

        # Slot-level importance scorer
        self.slot_importance = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Initialize gates conservatively: prefer retaining old knowledge
        nn.init.constant_(self.write_gate[-2].bias, bias)
        nn.init.constant_(self.forget_gate[-2].bias, -bias)

    def forward(self, old_memory, new_memory):
        """Apply gated update to memory.

        Args:
            old_memory: (B, num_slots, d_model) current memory state
            new_memory: (B, num_slots, d_model) proposed new memory state

        Returns:
            updated_memory: (B, num_slots, d_model) gated memory
            gate_stats: dict with gate activation statistics for monitoring
        """
        # Concatenate old and new for gating decisions
        combined = torch.cat([old_memory, new_memory], dim=-1)

        write_gate = self.write_gate(combined)
        forget_gate = self.forget_gate(combined)

        # Slot importance modulates the write strength
        importance = self.slot_importance(new_memory)

        # Gated update: balance old and new
        retained = forget_gate * old_memory
        accepted = write_gate * importance * new_memory
        updated_memory = retained + accepted

        # Collect statistics for monitoring
        gate_stats = {
            'write_gate_mean': write_gate.mean().item(),
            'forget_gate_mean': forget_gate.mean().item(),
            'importance_mean': importance.mean().item(),
            'memory_change': (updated_memory - old_memory).norm(dim=-1).mean().item(),
        }

        return updated_memory, gate_stats


class AdaptiveMemoryController(nn.Module):
    """Adaptive controller that modulates memory update intensity based on
    report quality and novelty.

    Args:
        d_model: Feature dimension
    """

    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model

        # Novelty detector: how different is this case from memory?
        self.novelty_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Update intensity modulator
        self.intensity_modulator = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_features, memory_features, current_loss=None):
        """Compute adaptive update intensity.

        Args:
            query_features: (B, d_model) current case representation
            memory_features: (B, d_model) retrieved memory representation
            current_loss: Optional current loss value per sample (B,)

        Returns:
            update_intensity: (B, 1) per-sample update strength
            novelty_score: (B, 1) novelty detection score
        """
        # Detect novelty: how different is query from retrieved memory?
        combined = torch.cat([query_features, memory_features], dim=-1)
        novelty_score = self.novelty_detector(combined)

        # Modulate update intensity
        if current_loss is not None:
            intensity_input = torch.cat([query_features, current_loss.unsqueeze(-1)], dim=-1)
        else:
            intensity_input = torch.cat([query_features, torch.zeros_like(novelty_score)], dim=-1)

        update_intensity = self.intensity_modulator(intensity_input)

        return update_intensity, novelty_score

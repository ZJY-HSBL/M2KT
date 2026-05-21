"""
Refiner Agent: Iterative report refinement through cross-attention between
draft report and visual features.

Mimics a radiologist's review process:
    1. Read draft report
    2. Re-examine image for each finding
    3. Identify sections that need revision
    4. Produce refined report

Architecture:
    Draft tokens -> Self-Attention -> Cross-Attention(Visual Feats) -> Refined tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from modules.Transformer import (
    MultiHeadedAttention, PositionwiseFeedForward,
    EncoderLayer, DecoderLayer, SublayerConnection,
    LayerNorm, clones, subsequent_mask,
)


class RefinerAgent(nn.Module):
    """Cross-modal refinement agent that improves draft reports using visual feedback.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_heads: Number of attention heads
        num_layers: Number of refinement layers
        dropout: Dropout rate
        max_refine_steps: Maximum number of refinement iterations
    """

    def __init__(self, d_model=512, d_ff=2048, num_heads=8, num_layers=3,
                 dropout=0.1, max_refine_steps=3):
        super().__init__()
        self.d_model = d_model
        self.max_refine_steps = max_refine_steps

        # Refinement encoder: process draft report
        attn = MultiHeadedAttention(num_heads, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.refine_encoder = clones(
            EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout),
            num_layers,
        )
        self.encoder_norm = LayerNorm(d_model)

        # Visual cross-attention: re-examine image regions
        self.visual_cross_attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.cross_attn_sublayer = SublayerConnection(d_model, dropout)
        self.cross_attn_ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.cross_attn_ff_sublayer = SublayerConnection(d_model, dropout)

        # Revision gate: predict which tokens need modification
        self.revision_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Stop criterion: predict whether refinement is sufficient
        self.stop_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, draft_tokens, visual_features, visual_mask=None, num_steps=None):
        """Refine draft report using visual features.

        Args:
            draft_tokens: (B, seq_len, d_model) draft report token embeddings
            visual_features: (B, num_patches, d_model) visual patch features
            visual_mask: Optional attention mask for visual features
            num_steps: Number of refinement steps (default: max_refine_steps)

        Returns:
            refined_tokens: (B, seq_len, d_model) refined report embeddings
            revision_mask: (B, seq_len) per-token revision probability
            stop_probs: (B,) probability of refinement being sufficient
        """
        if num_steps is None:
            num_steps = self.max_refine_steps

        refined = draft_tokens
        revision_masks = []
        stop_probs = []

        for step in range(num_steps):
            # Self-attention refinement
            refined = self.encoder_norm(self.refine_encoder[0](
                refined, mask=None
            ))

            # Cross-attend to visual features (re-examine the image)
            refined = self.cross_attn_sublayer(
                refined,
                lambda x: self.visual_cross_attn(x, visual_features, visual_features, visual_mask)
            )
            refined = self.cross_attn_ff_sublayer(refined, self.cross_attn_ff)

            # Predict revision needs
            rev_mask = self.revision_gate(refined)
            revision_masks.append(rev_mask)

            # Predict stop probability
            stop_prob = self.stop_predictor(refined.mean(dim=1))
            stop_probs.append(stop_prob)

            # Early stopping based on confidence
            if stop_prob.mean() > 0.95:
                break

        return refined, revision_masks, stop_probs


class DraftGenerator(nn.Module):
    """Initial draft report generator, compatible with standard Transformer decoder."""

    def __init__(self, d_model=512, d_ff=2048, num_heads=8, num_layers=6,
                 dropout=0.1, vocab_size=1000):
        super().__init__()
        attn = MultiHeadedAttention(num_heads, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.decoder_layers = clones(
            DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn),
                        copy.deepcopy(ff), dropout),
            num_layers,
        )
        self.norm = LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_emb, memory, src_mask, tgt_mask):
        x = tgt_emb
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        return self.output_proj(x)

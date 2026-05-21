"""
Temporal Difference Encoder: Encodes disease progression from paired
longitudinal chest X-rays.

Architecture:
    1. Siamese visual encoding: Both images share the same visual extractor
    2. Difference computation: Subtract prior features from current features
    3. Cross-temporal attention: Current attends to prior for change detection
    4. Progression classifier: Predicts which findings improved/worsened/stable

The encoder explicitly models three change states per finding:
    - Worsened: finding appeared or increased in severity
    - Improved: finding resolved or decreased in severity
    - Stable: finding unchanged between studies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.Transformer import MultiHeadedAttention, PositionwiseFeedForward, LayerNorm


class TemporalDifferenceEncoder(nn.Module):
    """Encodes changes between two time points in longitudinal chest X-rays.

    Args:
        d_model: Feature dimension
        num_labels: Number of CheXpert findings
        num_heads: Attention heads for cross-temporal attention
        dropout: Dropout rate
    """

    def __init__(self, d_model=512, num_labels=14, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels

        # Feature projections (shared for siamese encoding)
        self.proj_current = nn.Linear(d_model, d_model)
        self.proj_prior = nn.Linear(d_model, d_model)

        # Difference encoder
        self.diff_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Cross-temporal attention: current attends to prior
        self.cross_temporal_attn = MultiHeadedAttention(num_heads, d_model, dropout)

        # Temporal fusion
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # current + diff + attended
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Progression classifier: predicts change state per finding
        # Output: (B, num_labels, 3) for [stable, improved, worsened]
        self.progression_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_labels * 3),
        )

        # Time delta embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Severity change scorer
        self.severity_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_labels),
            nn.Tanh(),  # [-1, 1] where -1 = improved, +1 = worsened
        )

    def forward(self, current_features, prior_features, time_delta=None):
        """Encode temporal differences between paired images.

        Args:
            current_features: (B, d_model) features from current image
            prior_features: (B, d_model) features from prior image
            time_delta: (B,) normalized time delta in [0, 1]

        Returns:
            temporal_features: (B, d_model) encoded progression features
            progression_logits: (B, num_labels, 3) change state logits
            severity_change: (B, num_labels) continuous severity change [-1, 1]
            diff_features: (B, d_model) raw difference features
        """
        bsz = current_features.shape[0]

        # Project features (siamese-style)
        cur_proj = self.proj_current(current_features)
        pri_proj = self.proj_prior(prior_features)

        # Compute difference
        diff_features = cur_proj - pri_proj
        diff_encoded = self.diff_encoder(diff_features)

        # Cross-temporal attention: current queries prior
        cur_unsq = cur_proj.unsqueeze(1)  # (B, 1, d_model)
        pri_unsq = pri_proj.unsqueeze(1)  # (B, 1, d_model)
        attended = self.cross_temporal_attn(cur_unsq, pri_unsq, pri_unsq)
        attended = attended.squeeze(1)  # (B, d_model)

        # Time delta modulation
        if time_delta is not None:
            time_emb = self.time_embed(time_delta.unsqueeze(-1))
            diff_encoded = diff_encoded * (1.0 + time_emb)

        # Fuse all temporal signals
        temporal_features = self.temporal_fusion(
            torch.cat([cur_proj, diff_encoded, attended], dim=-1)
        )

        # Predict progression states
        prog_logits = self.progression_classifier(temporal_features)
        prog_logits = prog_logits.view(bsz, self.num_labels, 3)

        # Predict continuous severity change
        severity_change = self.severity_scorer(temporal_features)

        return temporal_features, prog_logits, severity_change, diff_features

    def get_change_description(self, progression_logits, severity_change, label_names=None):
        """Generate natural language description of disease progression.

        Args:
            progression_logits: (B, num_labels, 3) change state probabilities
            severity_change: (B, num_labels) continuous severity delta
            label_names: Optional list of finding names

        Returns:
            List of strings describing progression per sample
        """
        if label_names is None:
            label_names = [
                "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
                "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
                "Atelectasis", "Pneumothorax", "Pleural Effusion",
                "Pleural Thickening", "Fracture", "Support Device", "Enlarged",
            ]

        bsz = progression_logits.shape[0]
        prog_probs = F.softmax(progression_logits, dim=-1)  # (B, N, 3)
        change_states = ['stable', 'improved', 'worsened']

        descriptions = []
        for b in range(bsz):
            changes = []
            for n in range(len(label_names)):
                state = prog_probs[b, n].argmax().item()
                severity = severity_change[b, n].item()

                if state == 1 and severity < -0.3:  # Improved
                    changes.append(f"{label_names[n]} improved")
                elif state == 2 and severity > 0.3:  # Worsened
                    changes.append(f"{label_names[n]} worsened")
                elif state == 0:
                    changes.append(f"{label_names[n]} stable")

            if changes:
                desc = ". ".join(changes) + "."
            else:
                desc = "No significant interval change."

            descriptions.append(desc)

        return descriptions

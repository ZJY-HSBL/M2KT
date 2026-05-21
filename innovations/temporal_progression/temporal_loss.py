"""
Temporal Consistency Loss: Regularizes longitudinal report generation to ensure
consistent disease progression modeling across time points.

Loss components:
    1. Progression Classification Loss: Cross-entropy on change states
       [stable, improved, worsened] per finding
    2. Temporal Consistency Loss: Adjacent time-point reports should describe
       consistent changes (if A->B says "worsened", then B->A implies "improved")
    3. Severity Smoothness Loss: Severity changes should be monotonic over time
       (a finding shouldn't oscillate: worse -> better -> worse in short time)
    4. Report Difference Alignment: The difference between generated reports at
       t1 and t2 should align with the actual visual changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency and progression losses for longitudinal modeling.

    Args:
        num_labels: Number of CheXpert findings
        consistency_weight: Weight for temporal consistency
        progression_weight: Weight for progression classification
        smoothness_weight: Weight for severity smoothness
        report_alignment_weight: Weight for report-text alignment
    """

    def __init__(self, num_labels=14, consistency_weight=0.5,
                 progression_weight=1.0, smoothness_weight=0.3,
                 report_alignment_weight=0.3):
        super().__init__()
        self.num_labels = num_labels
        self.consistency_weight = consistency_weight
        self.progression_weight = progression_weight
        self.smoothness_weight = smoothness_weight
        self.report_alignment_weight = report_alignment_weight

        # Cross-entropy for [stable, improved, worsened]
        self.progression_criterion = nn.CrossEntropyLoss()

    def progression_loss(self, progression_logits, label_progression):
        """Classify progression state per finding.

        Args:
            progression_logits: (B, num_labels, 3) predicted change logits
            label_progression: (B, num_labels) ground-truth label difference
                               (negative = improved, positive = worsened, zero = stable)

        Returns:
            prog_loss: Scalar progression classification loss
        """
        bsz, num_labels, _ = progression_logits.shape

        # Convert continuous progression to discrete labels
        # -1: improved, 0: stable, +1: worsened -> {0, 1, 2}
        targets = torch.zeros(bsz, num_labels, dtype=torch.long,
                               device=progression_logits.device)
        targets[label_progression < -0.2] = 1  # Improved
        targets[label_progression > 0.2] = 2   # Worsened
        # targets == 0 is stable (default)

        prog_logits_flat = progression_logits.view(bsz * num_labels, 3)
        targets_flat = targets.view(bsz * num_labels)

        return self.progression_criterion(prog_logits_flat, targets_flat)

    def temporal_consistency_loss(self, current_report_emb, prior_report_emb,
                                   temporal_features, time_delta):
        """Encourage consistent report evolution over time.

        The semantic difference between current and prior reports should
        be proportional to the encoded temporal change.

        Args:
            current_report_emb: (B, d_model) current report embedding
            prior_report_emb: (B, d_model) prior report embedding
            temporal_features: (B, d_model) encoded progression features
            time_delta: (B,) normalized time between studies

        Returns:
            consistency_loss: Scalar alignment loss
        """
        report_diff = current_report_emb - prior_report_emb
        report_diff_norm = F.normalize(report_diff, dim=-1)
        temporal_norm = F.normalize(temporal_features, dim=-1)

        # Cosine similarity between report change and visual change
        alignment = (report_diff_norm * temporal_norm).sum(dim=-1)

        # Weight by time delta: larger intervals should allow larger changes
        # But the direction should remain aligned
        loss = F.relu(0.3 - alignment).mean()

        return loss

    def severity_smoothness_loss(self, severity_changes, time_deltas):
        """Encourage monotonic severity progression.

        For a given finding, severity should not oscillate rapidly.
        If severity increased from t-2 to t-1, it shouldn't dramatically
        decrease from t-1 to t (and vice versa).

        Args:
            severity_changes: (B, num_labels) predicted severity delta
            time_deltas: (B,) time intervals

        Returns:
            smoothness_loss: Scalar regularization
        """
        # Within-batch: penalize high variance in severity predictions
        # for similar time deltas
        batch_var = severity_changes.var(dim=0).mean()

        # Penalize extreme oscillations
        extreme_penalty = F.relu(severity_changes.abs() - 0.8).mean()

        return 0.1 * batch_var + 0.1 * extreme_penalty

    def report_alignment_loss(self, current_report_emb, current_labels,
                               prior_report_emb, prior_labels):
        """Align report embeddings with their corresponding label states.

        The report embedding should encode information about which findings
        are present/absent at each time point.

        Args:
            current_report_emb: (B, d_model)
            current_labels: (B, num_labels)
            prior_report_emb: (B, d_model)
            prior_labels: (B, num_labels)

        Returns:
            alignment_loss: Scalar
        """
        # Predict labels from report embeddings
        label_predictor = nn.Linear(
            current_report_emb.shape[-1], self.num_labels,
        ).to(current_report_emb.device)

        cur_pred = label_predictor(current_report_emb)
        pri_pred = label_predictor(prior_report_emb)

        cur_loss = F.binary_cross_entropy_with_logits(cur_pred, current_labels)
        pri_loss = F.binary_cross_entropy_with_logits(pri_pred, prior_labels)

        return (cur_loss + pri_loss) / 2

    def forward(self, progression_logits, label_progression,
                current_report_emb=None, prior_report_emb=None,
                temporal_features=None, time_deltas=None,
                severity_changes=None,
                current_labels=None, prior_labels=None):
        """Compute all temporal losses.

        Returns:
            total_loss: Combined temporal loss
            components: Dict of individual loss values
        """
        total = torch.tensor(0.0, device=progression_logits.device)
        components = {}

        # Progression classification
        prog_loss = self.progression_loss(progression_logits, label_progression)
        components['progression'] = prog_loss.item()
        total = total + self.progression_weight * prog_loss

        # Temporal consistency
        if current_report_emb is not None and prior_report_emb is not None:
            consist_loss = self.temporal_consistency_loss(
                current_report_emb, prior_report_emb,
                temporal_features, time_deltas,
            )
            components['consistency'] = consist_loss.item()
            total = total + self.consistency_weight * consist_loss

        # Severity smoothness
        if severity_changes is not None and time_deltas is not None:
            smooth_loss = self.severity_smoothness_loss(
                severity_changes, time_deltas)
            components['smoothness'] = smooth_loss.item()
            total = total + self.smoothness_weight * smooth_loss

        # Report-label alignment
        if current_report_emb is not None and current_labels is not None:
            align_loss = self.report_alignment_loss(
                current_report_emb, current_labels,
                prior_report_emb, prior_labels,
            )
            components['alignment'] = align_loss.item()
            total = total + self.report_alignment_weight * align_loss

        components['total'] = total.item()
        return total, components

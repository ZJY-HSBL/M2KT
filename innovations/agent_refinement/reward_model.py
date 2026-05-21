"""
Clinical Reward Model: Provides scalar reward signals for RL-based report
optimization, combining multiple clinical quality dimensions.

Reward components:
    1. NLG quality: BLEU-4, ROUGE-L, CIDEr scores vs ground truth
    2. Clinical accuracy: CheXpert label F1 between report and image labels
    3. Consistency: Inverse of contradiction score
    4. Completeness: Coverage of detected findings
    5. Factuality: Absence of hallucinated findings

The combined reward is used for Direct Preference Optimization (DPO) or
Reinforcement Learning from Clinical Feedback (RLCF).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalRewardModel(nn.Module):
    """Learned reward model for clinical report quality assessment.

    Combines automated metrics with learned quality dimensions to produce
    a scalar reward that correlates with radiologist preference.

    Args:
        d_model: Feature dimension
        num_labels: Number of CheXpert labels
        reward_dim: Internal reward representation dimension
    """

    def __init__(self, d_model=512, num_labels=14, reward_dim=256):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels

        # Reward component encoders
        self.quality_encoder = nn.Sequential(
            nn.Linear(d_model, reward_dim),
            nn.LayerNorm(reward_dim),
            nn.GELU(),
        )

        self.accuracy_encoder = nn.Sequential(
            nn.Linear(num_labels * 2, reward_dim),  # pred labels + gt labels
            nn.LayerNorm(reward_dim),
            nn.GELU(),
        )

        self.consistency_encoder = nn.Sequential(
            nn.Linear(d_model, reward_dim),
            nn.LayerNorm(reward_dim),
            nn.GELU(),
        )

        # Fusion and final reward prediction
        self.reward_head = nn.Sequential(
            nn.Linear(reward_dim * 3, reward_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(reward_dim, reward_dim // 2),
            nn.GELU(),
            nn.Linear(reward_dim // 2, 1),
            nn.Tanh(),  # Reward in [-1, 1]
        )

        # Bradley-Terry preference head for DPO
        self.preference_head = nn.Sequential(
            nn.Linear(reward_dim * 2, reward_dim),  # Compare two reports
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(reward_dim, 1),
            nn.Sigmoid(),  # P(report_A > report_B)
        )

    def forward(self, report_features, pred_labels, gt_labels, consistency_features):
        """Compute clinical reward for a generated report.

        Args:
            report_features: (B, d_model) report representation
            pred_labels: (B, num_labels) predicted CheXpert labels from image
            gt_labels: (B, num_labels) ground-truth CheXpert labels
            consistency_features: (B, d_model) features from consistency checker

        Returns:
            reward: (B,) scalar clinical quality reward in [-1, 1]
            reward_components: dict of individual reward components
        """
        # Quality: textual quality of generated report
        quality_rep = self.quality_encoder(report_features)

        # Accuracy: label prediction accuracy
        label_input = torch.cat([pred_labels, gt_labels], dim=-1)
        accuracy_rep = self.accuracy_encoder(label_input)

        # Consistency: internal coherence
        consistency_rep = self.consistency_encoder(consistency_features)

        # Fuse and predict reward
        combined = torch.cat([quality_rep, accuracy_rep, consistency_rep], dim=-1)
        reward = self.reward_head(combined).squeeze(-1)

        components = {
            'quality': quality_rep.norm(dim=-1),
            'accuracy': accuracy_rep.norm(dim=-1),
            'consistency': consistency_rep.norm(dim=-1),
        }

        return reward, components

    def predict_preference(self, report_a_features, report_b_features):
        """Predict which of two reports is preferred (for DPO training).

        Args:
            report_a_features: (B, d_model) first report representation
            report_b_features: (B, d_model) second report representation

        Returns:
            p_a_better: (B,) probability that report A is preferred over B
        """
        combined = torch.cat([report_a_features, report_b_features], dim=-1)
        return self.preference_head(combined).squeeze(-1)


class CompositeReward:
    """Non-learned composite reward combining multiple heuristic signals.

    Used alongside the learned reward model for stable training.
    """

    def __init__(self, w_nlg=1.0, w_clinical=2.0, w_consistency=1.5, w_completeness=1.5):
        self.w_nlg = w_nlg
        self.w_clinical = w_clinical
        self.w_consistency = w_consistency
        self.w_completeness = w_completeness

    def compute(self, nlg_scores, clinical_f1, consistency_score, completeness_score):
        """Compute composite reward from individual components.

        Args:
            nlg_scores: dict with keys 'BLEU_4', 'ROUGE_L', 'CIDEr'
            clinical_f1: float tensor (B,) label prediction F1
            consistency_score: float tensor (B,)
            completeness_score: float tensor (B,)

        Returns:
            total_reward: float tensor (B,)
        """
        nlg_reward = (nlg_scores.get('BLEU_4', 0) + nlg_scores.get('ROUGE_L', 0)) / 2.0
        total = (
            self.w_nlg * nlg_reward +
            self.w_clinical * clinical_f1 +
            self.w_consistency * consistency_score +
            self.w_completeness * completeness_score
        )
        return total / (self.w_nlg + self.w_clinical + self.w_consistency + self.w_completeness)

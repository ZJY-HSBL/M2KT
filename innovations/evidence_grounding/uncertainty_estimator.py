"""
Uncertainty Estimator: Quantifies model confidence for each generated finding
using Monte Carlo Dropout and Deep Ensemble methods.

Two types of uncertainty:
    1. Aleatoric uncertainty: Inherent noise in the data (image quality, ambiguous findings)
    2. Epistemic uncertainty: Model uncertainty due to limited knowledge

Outputs:
    - Per-finding confidence scores
    - Per-sentence uncertainty intervals
    - Overall report reliability score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UncertaintyEstimator(nn.Module):
    """Quantifies prediction uncertainty through MC Dropout sampling.

    Args:
        d_model: Feature dimension
        num_labels: Number of findings
        mc_samples: Number of MC Dropout forward passes
        dropout_rate: Dropout rate for MC sampling
    """

    def __init__(self, d_model=512, num_labels=14, mc_samples=10, dropout_rate=0.2):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.mc_samples = mc_samples

        # Aleatoric uncertainty head: predicts variance per finding
        self.aleatoric_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_labels),
            nn.Softplus(),  # Variance must be positive
        )

        # Epistemic uncertainty aggregator
        self.epistemic_aggregator = nn.Sequential(
            nn.Linear(num_labels * 2, d_model // 2),  # mean + variance
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Reliability scorer
        self.reliability_scorer = nn.Sequential(
            nn.Linear(d_model + num_labels + 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def enable_mc_dropout(self, model):
        """Enable dropout layers at test time for MC sampling."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def mc_forward(self, model, images, num_samples=None):
        """Perform multiple stochastic forward passes for uncertainty estimation.

        Args:
            model: The main model (with dropout enabled at test time)
            images: (B, C, H, W) input
            num_samples: Number of MC samples (default: self.mc_samples)

        Returns:
            predictions: List of (B, ...) outputs per sample
        """
        if num_samples is None:
            num_samples = self.mc_samples

        self.enable_mc_dropout(model)

        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(images, mode='sample')
                predictions.append(output)

        return predictions

    def compute_aleatoric(self, features):
        """Compute aleatoric (data) uncertainty from features.

        Args:
            features: (B, d_model) image/report features

        Returns:
            aleatoric_var: (B, num_labels) predicted variance per finding
        """
        return self.aleatoric_head(features)

    def compute_epistemic(self, mc_predictions):
        """Compute epistemic (model) uncertainty from MC samples.

        Uses predictive entropy across MC samples:
            H[y|x] ≈ -Σ p(y|x, w_i) log p(y|x, w_i) / M

        Args:
            mc_predictions: List of T tensors, each (B, ...)

        Returns:
            epistemic_uncertainty: (B,) per-sample epistemic uncertainty
            predictive_mean: (B, ...) average prediction across samples
            predictive_variance: (B, ...) variance across samples
        """
        # Stack predictions: (M, B, ...)
        stacked = torch.stack(mc_predictions, dim=0)

        predictive_mean = stacked.mean(dim=0)
        predictive_variance = stacked.var(dim=0)

        # Predictive entropy
        eps = 1e-8
        entropy_per_sample = -(
            predictive_mean * torch.log(predictive_mean + eps)
        ).sum(dim=-1)
        epistemic_uncertainty = entropy_per_sample.mean(dim=-1)  # (B,)

        return epistemic_uncertainty, predictive_mean, predictive_variance

    def compute_reliability(self, features, vis_labels, aleatoric_var, epistemic_score):
        """Compute overall report reliability score.

        Args:
            features: (B, d_model) global features
            vis_labels: (B, num_labels) predicted labels
            aleatoric_var: (B, num_labels) aleatoric variance
            epistemic_score: (B,) epistemic uncertainty

        Returns:
            reliability: (B,) reliability score in [0, 1]
        """
        aleatoric_summary = aleatoric_var.mean(dim=-1, keepdim=True)
        epistemic = epistemic_score.unsqueeze(-1)

        combined = torch.cat([
            features,
            vis_labels,
            aleatoric_summary,
            epistemic,
        ], dim=-1)

        return self.reliability_scorer(combined).squeeze(-1)

    def forward(self, model, images, features, vis_labels, num_mc_samples=None):
        """Full uncertainty estimation pipeline.

        Args:
            model: Main model
            images: (B, C, H, W) input
            features: (B, d_model) extracted features
            vis_labels: (B, num_labels) predicted labels
            num_mc_samples: MC samples (optional)

        Returns:
            Dict with all uncertainty estimates
        """
        # Aleatoric uncertainty
        aleatoric_var = self.compute_aleatoric(features)

        # Epistemic uncertainty via MC Dropout
        mc_preds = self.mc_forward(model, images, num_mc_samples)
        epistemic_score, pred_mean, pred_var = self.compute_epistemic(mc_preds)

        # Overall reliability
        reliability = self.compute_reliability(
            features, vis_labels, aleatoric_var, epistemic_score)

        # Per-finding confidence intervals
        confidence = 1.0 / (1.0 + aleatoric_var.mean(dim=-1))

        return {
            'aleatoric_var': aleatoric_var,
            'epistemic_score': epistemic_score,
            'predictive_mean': pred_mean,
            'predictive_variance': pred_var,
            'reliability': reliability,
            'confidence': confidence,
        }

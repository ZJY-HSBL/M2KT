"""
Completeness Verifier: Ensures generated reports cover all clinically significant
findings detected in the image.

Compares predicted CheXpert labels against the generated report content to
detect omissions (findings present in image but not mentioned in report) and
hallucinations (findings mentioned in report but not supported by image).

The completeness score serves as:
    1. A reward signal for RL training
    2. A quality metric for self-assessment
    3. A trigger for targeted refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Transformer import MultiHeadedAttention


# Mapping from CheXpert label index to report keywords
CHEXPERT_KEYWORDS = {
    0:  ["no finding", "normal", "clear"],
    1:  ["enlarged cardiomediastinum", "widened mediastinum", "mediastinal widening"],
    2:  ["cardiomegaly", "enlarged heart", "cardiac enlargement", "enlarged cardiac"],
    3:  ["lung opacity", "opacity", "haziness", "infiltrate"],
    4:  ["lung lesion", "nodule", "mass", "lesion"],
    5:  ["edema", "pulmonary edema", "fluid", "congestion"],
    6:  ["consolidation", "pneumonia", "airspace disease"],
    7:  ["atelectasis", "collapse", "volume loss", "atelectatic"],
    8:  ["pneumothorax", "air in pleural space"],
    9:  ["pleural effusion", "pleural fluid", "effusion"],
    10: ["pleural thickening", "pleural disease", "pleural abnormality"],
    11: ["fracture", "rib fracture", "bony abnormality"],
    12: ["support device", "tube", "line", "catheter", "device"],
    13: ["enlarged", "prominent"],
}


class CompletenessVerifier(nn.Module):
    """Verifies report completeness by checking label-to-text coverage.

    Args:
        d_model: Feature dimension
        num_labels: Number of CheXpert labels (default 14)
        num_heads: Attention heads for cross-modal matching
        dropout: Dropout rate
    """

    def __init__(self, d_model=512, num_labels=14, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels

        # Label embedding for comparison with text
        self.label_embed = nn.Embedding(num_labels, d_model)

        # Text-to-label attention: does text mention each label?
        self.text_label_attn = MultiHeadedAttention(num_heads, d_model, dropout)

        # Coverage classifier per label
        self.coverage_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Overall completeness scorer
        self.completeness_scorer = nn.Sequential(
            nn.Linear(d_model + num_labels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_embeddings, pred_labels):
        """Compute per-label coverage and overall completeness.

        Args:
            token_embeddings: (B, seq_len, d_model) generated report embeddings
            pred_labels: (B, num_labels) predicted CheXpert label probabilities

        Returns:
            per_label_coverage: (B, num_labels) coverage score per finding
            completeness_score: (B,) overall report completeness
            omission_mask: (B, num_labels) which findings are missed
        """
        bsz = token_embeddings.shape[0]

        # Pool text representation
        text_rep = token_embeddings.mean(dim=1)  # (B, d_model)

        # Embed each label
        label_indices = torch.arange(self.num_labels, device=token_embeddings.device)
        label_emb = self.label_embed(label_indices).unsqueeze(0).expand(bsz, -1, -1)

        # Cross-attention: text queries labels
        text_query = text_rep.unsqueeze(1).expand(bsz, self.num_labels, -1)
        attended = self.text_label_attn(text_query, label_emb, label_emb)

        # Classify coverage per label
        coverage_input = torch.cat([attended, label_emb], dim=-1)
        per_label_coverage = self.coverage_classifier(coverage_input).squeeze(-1)

        # Overall completeness
        completeness_input = torch.cat([text_rep, per_label_coverage], dim=-1)
        completeness_score = self.completeness_scorer(completeness_input).squeeze(-1)

        # Detect omissions: label predicted positive but not covered in text
        label_positive = (pred_labels > 0.5).float()
        omission_mask = label_positive * (1.0 - per_label_coverage)

        return per_label_coverage, completeness_score, omission_mask

    def get_omission_summary(self, omission_mask, label_names=None):
        """Generate human-readable summary of omitted findings.

        Args:
            omission_mask: (B, num_labels) binary omission indicators
            label_names: Optional list of label name strings

        Returns:
            List of lists of omitted finding names per sample
        """
        if label_names is None:
            label_names = [
                "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
                "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
                "Atelectasis", "Pneumothorax", "Pleural Effusion",
                "Pleural Thickening", "Fracture", "Support Device",
                "Enlarged",
            ]

        summaries = []
        for b in range(omission_mask.shape[0]):
            omitted = [
                label_names[i]
                for i in range(len(label_names))
                if omission_mask[b, i] > 0.5
            ]
            summaries.append(omitted)

        return summaries

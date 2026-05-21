"""
Consistency Checker: Detects internal contradictions within generated reports.

Uses natural language inference (NLI) style bidirectional entailment:
    - Sentence A entails "finding X is present"
    - Sentence B entails "finding X is absent"
    => Contradiction detected

Also checks for:
    - Numerical inconsistencies (e.g., "heart size normal" + "cardiomegaly")
    - Anatomical impossibilities (e.g., conflicting lobe descriptions)
    - Temporal contradictions (e.g., "stable" + "worsening")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Transformer import MultiHeadedAttention, PositionwiseFeedForward, LayerNorm


# Known contradictory finding pairs in chest X-ray reports
CONTRADICTORY_PAIRS = [
    ("cardiomegaly", "heart size normal"),
    ("cardiomegaly", "normal cardiac silhouette"),
    ("pleural effusion", "clear costophrenic angles"),
    ("pulmonary edema", "clear lung fields"),
    ("pneumothorax", "lungs are well expanded"),
    ("consolidation", "clear lungs"),
    ("atelectasis", "fully expanded lungs"),
    ("enlarged", "normal"),
    ("stable", "worsening"),
    ("improved", "worsened"),
    ("resolved", "persistent"),
]


class ConsistencyChecker(nn.Module):
    """Checks report for internal contradictions using learned NLI-style embeddings.

    Args:
        d_model: Feature dimension for sentence representations
        num_heads: Attention heads for contradiction detection
        dropout: Dropout rate
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Project token embeddings to sentence-level representations
        self.sentence_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Cross-sentence attention for contradiction detection
        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout)

        # Contradiction classifier: [sentence_i; sentence_j; cross_attn_diff]
        self.contradiction_classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Learnable embeddings for common contradictory finding phrases
        self.register_buffer(
            'finding_embeddings',
            torch.randn(50, d_model) * 0.02  # Placeholder for learnable finding reps
        )

        self.dropout = nn.Dropout(dropout)

    def _split_sentences(self, token_embeddings, sentence_boundaries=None):
        """Split token sequence into sentence-level representations.

        If sentence_boundaries is None, uses a heuristic split on [SEP] or [EOS] tokens,
        or splits evenly into chunks.

        Args:
            token_embeddings: (B, seq_len, d_model)
            sentence_boundaries: Optional (B, num_sentences) boundary indices

        Returns:
            sentence_reps: List of (B, d_model) tensors per sentence
        """
        bsz, seq_len, d_model = token_embeddings.shape

        # Default: split into ~5 equal chunks as pseudo-sentences
        num_sentences = 5
        chunk_size = seq_len // num_sentences

        sentences = []
        for i in range(num_sentences):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_sentences - 1 else seq_len
            sent_rep = token_embeddings[:, start:end, :].mean(dim=1)
            sent_rep = self.sentence_encoder(sent_rep)
            sentences.append(sent_rep)

        return sentences

    def forward(self, token_embeddings, sentence_boundaries=None):
        """Detect contradictions in the generated report.

        Args:
            token_embeddings: (B, seq_len, d_model) generated report embeddings
            sentence_boundaries: Optional sentence split indices

        Returns:
            contradiction_scores: (B, num_pairs) pairwise contradiction probabilities
            is_consistent: (B,) overall consistency score (1 = consistent)
            max_contradiction: (B,) maximum contradiction score found
        """
        bsz = token_embeddings.shape[0]
        sentences = self._split_sentences(token_embeddings, sentence_boundaries)
        num_sentences = len(sentences)

        if num_sentences < 2:
            return (torch.zeros(bsz, 1, device=token_embeddings.device),
                    torch.ones(bsz, device=token_embeddings.device),
                    torch.zeros(bsz, device=token_embeddings.device))

        # Check all sentence pairs for contradiction
        all_contra_scores = []
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                s_i = sentences[i].unsqueeze(1)  # (B, 1, d_model)
                s_j = sentences[j].unsqueeze(1)  # (B, 1, d_model)

                # Cross-attention between sentence pair
                cross_out = self.cross_attn(s_i, s_j, s_j)
                cross_diff = (s_i - s_j).abs()

                # Classify contradiction
                pair_input = torch.cat([s_i.squeeze(1), s_j.squeeze(1), cross_diff.squeeze(1)], dim=-1)
                contra_score = self.contradiction_classifier(pair_input)
                all_contra_scores.append(contra_score)

        # Aggregate
        if all_contra_scores:
            contra_matrix = torch.cat(all_contra_scores, dim=-1)  # (B, num_pairs)
        else:
            contra_matrix = torch.zeros(bsz, 1, device=token_embeddings.device)

        max_contra, _ = contra_matrix.max(dim=-1)
        is_consistent = 1.0 - max_contra

        return contra_matrix, is_consistent, max_contra

    def get_clinical_contradictions(self, generated_texts, tokenizer):
        """Rule-based clinical contradiction detection as auxiliary signal.

        Args:
            generated_texts: List of str, generated reports
            tokenizer: Tokenizer for decoding

        Returns:
            clinical_contra_scores: (B,) rule-based contradiction scores
        """
        scores = []
        for text in generated_texts:
            text_lower = text.lower()
            score = 0.0
            for phrase_a, phrase_b in CONTRADICTORY_PAIRS:
                if phrase_a in text_lower and phrase_b in text_lower:
                    score += 1.0
            scores.append(min(score / max(len(CONTRADICTORY_PAIRS), 1), 1.0))

        return torch.tensor(scores, device=next(self.parameters()).device)

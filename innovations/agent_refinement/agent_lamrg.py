"""
AgentLAMRG: Agent-based iterative self-refinement model for radiology report generation.

Four-stage pipeline:
    1. Base Generator: Produce initial draft report from images
    2. Consistency Checker: Detect internal contradictions
    3. Completeness Verifier: Identify missing findings
    4. Refiner Agent: Revise report based on feedback from (2) and (3)

The model can iterate up to K times, with each iteration producing an
improved report. DPO training is used to align refinement behavior with
clinical quality preferences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder

from .refiner_agent import RefinerAgent
from .consistency_checker import ConsistencyChecker
from .completeness_verifier import CompletenessVerifier


class AgentLAMRG(nn.Module):
    """Agent-based LAMRG with iterative self-refinement.

    Args:
        args: Configuration namespace
        tokenizer: Report tokenizer
    """

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_labels = args.num_labels
        self.d_vf = args.d_vf
        self.dropout = args.dropout
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.max_refine_steps = getattr(args, 'max_refine_steps', 3)

        # ---- Stage 0: Visual Extractor ----
        self.visual_extractor = VisualExtractor(args)

        # ---- Stage 1: Base Generator ----
        # Simple linear projection from d_vf to d_model
        self.att_proj = nn.Linear(self.d_vf, self.d_model)
        self.fc_proj = nn.Linear(self.d_vf, self.d_model)

        self.encoder_decoder = TransformerModel(args, tokenizer)

        # ---- Projections for labels and features ----
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_labels)

        # ---- Stage 2: Consistency Checker ----
        self.consistency_checker = ConsistencyChecker(
            d_model=self.d_model,
            num_heads=args.num_heads,
            dropout=self.dropout,
        )

        # ---- Stage 3: Completeness Verifier ----
        self.completeness_verifier = CompletenessVerifier(
            d_model=self.d_model,
            num_labels=self.num_labels,
            num_heads=args.num_heads,
            dropout=self.dropout,
        )

        # ---- Stage 4: Refiner Agent ----
        self.refiner = RefinerAgent(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_heads=args.num_heads,
            num_layers=getattr(args, 'refiner_layers', 3),
            dropout=self.dropout,
            max_refine_steps=self.max_refine_steps,
        )

        # ---- Feedback fusion for refinement conditioning ----
        self.feedback_fusion = nn.Sequential(
            nn.Linear(self.d_model + self.num_labels + 1, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.att_proj, self.fc_proj, self.linear_z, self.classifier]:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward_iu_xray(self, images):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        out_labels = labels_0
        return att_feats, fc_feats, out_labels

    def forward_mimic_cxr(self, images):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        return att_feats, fc_feats, out_labels

    def generate_draft(self, att_feats, fc_feats, targets=None, mode='sample'):
        """Stage 1: Generate initial draft report."""
        if mode == 'train' and targets is not None:
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        else:
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        return output

    def get_report_embeddings(self, output_logits):
        """Convert output logits to continuous embeddings for refinement.

        Uses soft embeddings: weighted sum of token embeddings by predicted probs.
        """
        probs = F.softmax(output_logits, dim=-1)  # (B, seq, vocab)
        weight = self.encoder_decoder.model.tgt_embed[0].lut.weight  # (vocab, d_model)
        emb = torch.matmul(probs, weight)  # (B, seq, d_model)
        return emb

    def forward(self, images, targets=None, labels=None, mode='train'):
        """Full agent pipeline: generate -> check -> verify -> refine.

        Args:
            images: Input chest X-ray images
            targets: Ground-truth report tokens (training only)
            labels: CheXpert labels
            mode: 'train' or 'sample'

        Returns:
            Dictionary containing all outputs at each stage
        """
        bsz = images.shape[0]

        # ---- Visual Feature Extraction ----
        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, fc_feats, vis_labels_raw = ve(images)

        # Project features
        att_feats_proj = self.att_proj(att_feats)
        fc_feats_proj = self.fc_proj(fc_feats)

        z_img = self.linear_z(fc_feats)
        vis_labels = self.classifier(z_img)

        # ---- Stage 1: Draft Generation ----
        draft_output = self.generate_draft(
            fc_feats, att_feats,
            targets=targets if mode == 'train' else None,
            mode='forward' if mode == 'train' else 'sample',
        )

        # Get draft embeddings for consistency/refinement
        if mode == 'train':
            draft_emb = self.encoder_decoder.model.tgt_embed(targets)
        else:
            draft_emb = self.get_report_embeddings(draft_output)

        # ---- Stage 2: Consistency Check ----
        contra_scores, consistency_score, max_contra = self.consistency_checker(draft_emb)

        # ---- Stage 3: Completeness Verifier ----
        per_label_coverage, completeness_score, omission_mask = \
            self.completeness_verifier(draft_emb, vis_labels)

        # ---- Stage 4: Refinement ----
        # Build feedback conditioning
        feedback = torch.cat([
            draft_emb.mean(dim=1),
            omission_mask,
            consistency_score.unsqueeze(-1),
        ], dim=-1)
        feedback_cond = self.feedback_fusion(feedback).unsqueeze(1)

        # Refine draft with visual features and feedback
        draft_with_feedback = draft_emb + feedback_cond
        refined_emb, revision_masks, stop_probs = self.refiner(
            draft_with_feedback, att_feats_proj,
        )

        # Decode refined embeddings to logits
        refined_logits = F.log_softmax(
            self.encoder_decoder.logit(refined_emb), dim=-1
        )

        results = {
            'draft_output': draft_output,
            'refined_output': refined_logits,
            'vis_labels': vis_labels,
            'z_img': z_img,
            'consistency_score': consistency_score,
            'completeness_score': completeness_score,
            'omission_mask': omission_mask,
            'contra_scores': contra_scores,
            'revision_masks': revision_masks,
            'stop_probs': stop_probs,
        }

        return results

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

"""
TemporalLAMRG: Longitudinal radiology report generation with disease progression
modeling.

Takes a pair of chest X-rays from different time points and generates a report
that describes:
    1. Current findings (what is visible now)
    2. Interval changes (what has changed since the prior study)
    3. Disease progression (improved/worsened/stable per finding)

Architecture:
    [Current Image] ----+---> [Visual Extractor] ---> [Current Features]
                         |                                    |
    [Prior Image]  -----+                                     |
                         |                                    v
                         +---> [Visual Extractor] ---> [Prior Features]
                              (shared weights)                |
                                                              v
                                              [Temporal Difference Encoder]
                                                              |
                                              [Progression Features]
                                                              |
                                                              v
                                              [LAMRG Decoder] -> Report
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder, MHA_FF
from .temporal_encoder import TemporalDifferenceEncoder


class TemporalLAMRG(nn.Module):
    """Longitudinal LAMRG for paired temporal chest X-ray analysis.

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
        self.num_slots = getattr(args, 'num_slots', 64)
        self.d_vf = args.d_vf
        self.dropout = args.dropout
        self.tgt_vocab = len(tokenizer.idx2token) + 1

        # ---- Shared Visual Extractor (Siamese) ----
        self.visual_extractor = VisualExtractor(args)

        # ---- Text Encoder ----
        self.txt_encoder = TextEncoder(
            self.d_model, self.d_ff, args.num_layers,
            self.tgt_vocab, self.num_labels,
            args.num_heads, self.dropout,
        )

        # ---- Knowledge Base Memory ----
        self.update_memory = MHA_FF(
            self.d_model, self.d_ff,
            getattr(args, 'num_memory_heads', 8), self.dropout,
        )
        self.select_prior = MHA_FF(
            self.d_model, self.d_ff,
            getattr(args, 'num_memory_heads', 8), self.dropout,
        )
        self.memory, self.mask = self._init_memory()
        self.get_mem = nn.Linear(self.d_model, self.d_model)

        # ---- Temporal Difference Encoder ----
        self.temporal_encoder = TemporalDifferenceEncoder(
            d_model=self.d_model,
            num_labels=self.num_labels,
            num_heads=args.num_heads,
            dropout=self.dropout,
        )

        # ---- Projections ----
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)
        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        # ---- Temporal feature projection ----
        self.temporal_proj = nn.Linear(self.d_model, self.d_model)

        # ---- Transformer Encoder-Decoder ----
        self.encoder_decoder = TransformerModel(args, tokenizer)

        # ---- Progression-conditioned token embedding ----
        self.progression_embed = nn.Sequential(
            nn.Linear(self.num_labels * 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.linear_z, self.get_mem, self.linear_feat,
                       self.classifier, self.embed_labels, self.temporal_proj]:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def _init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def encode_image(self, images):
        """Encode a single image through the visual extractor."""
        if images.dim() == 5:  # Multi-view (IU format)
            att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
            att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])
            fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1]), dim=0)
            att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
            out_labels = torch.mean(torch.stack([labels_0, labels_1]), dim=0)
        else:
            att_feats, fc_feats, out_labels = self.visual_extractor(images)

        return att_feats, fc_feats, out_labels

    def forward(self, current_images, prior_images=None, targets=None,
                labels=None, time_deltas=None, mode='train'):
        """Forward with temporal modeling.

        Args:
            current_images: (B, C, H, W) current chest X-rays
            prior_images: (B, C, H, W) prior chest X-rays (optional)
            targets: (B, seq_len) current report token ids
            labels: (B, num_labels) current CheXpert labels
            time_deltas: (B,) normalized time deltas
            mode: 'train' or 'sample'

        Returns:
            train: (output, vis_labels, txt_labels, z_img, z_txt,
                    progression_logits, severity_change, temporal_features)
            sample: (output, vis_labels, progression_logits, severity_change)
        """
        bsz = current_images.shape[0]

        # ---- Encode Current Image ----
        cur_att_feats, cur_fc_feats, cur_vis_labels = self.encode_image(current_images)

        # ---- Encode Prior Image (Siamese) ----
        if prior_images is None:
            prior_images = current_images

        pri_att_feats, pri_fc_feats, pri_vis_labels = self.encode_image(prior_images)

        # ---- Global Projections ----
        z_img_cur = self.linear_z(cur_fc_feats)
        z_img_pri = self.linear_z(pri_fc_feats)

        vis_labels = self.classifier(z_img_cur)

        # ---- Temporal Encoding ----
        temporal_features, progression_logits, severity_change, diff_features = \
            self.temporal_encoder(z_img_cur, z_img_pri, time_deltas)

        # ---- Knowledge Base Retrieval ----
        memory = self.get_mem(self.memory.to(current_images)).expand(bsz, -1, -1)
        mask = self.mask.to(current_images).expand(bsz, -1, -1)

        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(memory, txt_feats, mask)
        else:
            txt_feats, z_txt, txt_labels = None, None, None

        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        memory_prior = self.select_prior(emb_labels, memory)

        # ---- Augment features with temporal information ----
        temporal_proj = self.temporal_proj(temporal_features).unsqueeze(1)
        att_feats = torch.cat((
            cur_att_feats,
            self.linear_feat(memory_prior),
            temporal_proj,
        ), dim=1)

        # ---- Decoding ----
        if mode == 'train':
            output = self.encoder_decoder(cur_fc_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img_cur, z_txt,
                    progression_logits, severity_change, temporal_features)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(cur_fc_feats, att_feats, opt=self.args, mode='sample')
            return (output, vis_labels, progression_logits, severity_change)
        else:
            raise ValueError

    def generate_progression_report(self, current_images, prior_images, time_deltas=None):
        """Generate a report with explicit progression description.

        Args:
            current_images: Current chest X-rays
            prior_images: Prior chest X-rays
            time_deltas: Time intervals

        Returns:
            report: Generated report text
            progression_description: Human-readable change description
        """
        self.eval()
        with torch.no_grad():
            output, vis_labels, prog_logits, sev_change = self.forward(
                current_images, prior_images, time_deltas=time_deltas, mode='sample',
            )

        report = self.tokenizer.decode_batch(output.cpu().numpy())
        progression_desc = self.temporal_encoder.get_change_description(
            prog_logits, sev_change,
        )

        return report, progression_desc

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

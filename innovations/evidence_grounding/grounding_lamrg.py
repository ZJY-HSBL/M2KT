"""
GroundingLAMRG: Explainable radiology report generation with evidence grounding.

Extends LAMRG v12 with:
    1. Spatial attention extraction from decoder cross-attention
    2. Sentence-to-region grounding (which image region supports each sentence)
    3. Per-finding uncertainty quantification
    4. Report reliability scoring

Outputs:
    - Generated report text
    - Spatial attention heatmaps (B, num_sentences, H, W)
    - Finding confidence scores (B, num_labels)
    - Report reliability score (B,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder, MHA_FF
from .spatial_attention import SpatialAttentionHead
from .uncertainty_estimator import UncertaintyEstimator


class GroundingLAMRG(nn.Module):
    """Explainable report generation model with spatial evidence grounding.

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
        self.image_size = getattr(args, 'image_size', (224, 224))

        # ---- Visual Extractor ----
        self.visual_extractor = VisualExtractor(args)

        # ---- Text Encoder for knowledge base ----
        self.txt_encoder = TextEncoder(
            self.d_model, self.d_ff, args.num_layers,
            self.tgt_vocab, self.num_labels,
            args.num_heads, self.dropout,
        )

        # ---- Knowledge Base Memory (from LAMRG v12) ----
        self.update_memory = MHA_FF(
            self.d_model, self.d_ff,
            getattr(args, 'num_memory_heads', 8), self.dropout,
        )
        self.select_prior = MHA_FF(
            self.d_model, self.d_ff,
            getattr(args, 'num_memory_heads', 8), self.dropout,
        )
        self.memory, self.mask = self._init_memory()

        # ---- Projections ----
        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)
        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        # ---- Transformer Encoder-Decoder ----
        self.encoder_decoder = TransformerModel(args, tokenizer)

        # ---- Spatial Attention Module ----
        patch_size = getattr(args, 'patch_size', 32)
        self.spatial_attention = SpatialAttentionHead(
            d_model=self.d_model,
            num_heads=args.num_heads,
            image_size=self.image_size,
            patch_size=patch_size,
        )

        # ---- Uncertainty Estimator ----
        self.uncertainty_estimator = UncertaintyEstimator(
            d_model=self.d_model,
            num_labels=self.num_labels,
            mc_samples=getattr(args, 'mc_samples', 10),
        )

        # ---- Finding Embeddings for anatomical region mapping ----
        self.finding_embed = nn.Embedding(self.num_labels, self.d_model)

        self._init_weights()

    def _init_weights(self):
        for module in [self.linear_z, self.get_mem, self.linear_feat,
                       self.classifier, self.embed_labels]:
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

    def forward_iu_xray(self, images):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1]), dim=0)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        out_labels = torch.mean(torch.stack([labels_0, labels_1]), dim=0)
        return att_feats, fc_feats, out_labels

    def forward_mimic_cxr(self, images):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        return att_feats, fc_feats, out_labels

    def forward(self, images, targets=None, labels=None, mode='train'):
        """Forward with evidence grounding outputs.

        Returns (in addition to standard outputs):
            spatial_maps: (B, tgt_len, H, W) per-token attention heatmaps
            uncertainty: Dict with aleatoric/epistemic/reliability scores
            cross_attn_weights: Raw cross-attention for analysis
        """
        bsz = images.shape[0]

        # ---- Visual feature extraction ----
        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        # ---- Global projections ----
        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        # ---- Memory retrieval ----
        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        mask = self.mask.to(images).expand(bsz, -1, -1)

        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(memory, txt_feats, mask)
        else:
            txt_feats, z_txt, txt_labels = None, None, None

        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        prior = self.select_prior(emb_labels, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        # ---- Decoding ----
        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError

        # ---- Extract spatial attention from decoder cross-attention ----
        cross_attn_weights = self._get_cross_attention_weights()
        spatial_maps, sentence_maps, finding_scores = self.spatial_attention(
            cross_attn_weights,
        )

        # ---- Uncertainty estimation ----
        uncertainty = None
        if mode == 'sample':
            uncertainty = self.uncertainty_estimator(
                self, images, z_img, vis_labels,
            )

        results = {
            'output': output,
            'vis_labels': vis_labels,
            'z_img': z_img,
            'spatial_maps': spatial_maps,
            'sentence_maps': sentence_maps,
            'finding_scores': finding_scores,
            'cross_attn_weights': cross_attn_weights,
            'uncertainty': uncertainty,
        }

        if mode == 'train':
            results['txt_labels'] = txt_labels
            results['z_txt'] = z_txt

        return results

    def _get_cross_attention_weights(self):
        """Extract cross-attention weights from the decoder's last layer."""
        try:
            decoder = self.encoder_decoder.model.decoder
            last_layer = decoder.layers[-1]
            attn_weights = last_layer.src_attn.attn
            return attn_weights
        except Exception:
            return torch.zeros(
                1, self.args.num_heads, 1, 1,
                device=next(self.parameters()).device,
            )

    def generate_with_evidence(self, images):
        """Generate report with evidence maps for clinical review.

        Args:
            images: (B, C, H, W) input chest X-rays

        Returns:
            Dict with report text, spatial evidence maps, confidence scores
        """
        self.eval()
        with torch.no_grad():
            results = self.forward(images, mode='sample')

        # Decode report
        report_texts = self.tokenizer.decode_batch(
            results['output'].cpu().numpy()
        )

        # Uncertainty
        uncertainty = self.uncertainty_estimator(
            self, images, results['z_img'], results['vis_labels'],
        )

        return {
            'reports': report_texts,
            'spatial_evidence': results['sentence_maps'],
            'finding_confidence': 1.0 / (1.0 + uncertainty['aleatoric_var']),
            'reliability': uncertainty['reliability'],
        }

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

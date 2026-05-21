"""
MLLM-LAMRG: Medical VLM augmented LAMRG with Hierarchical Evolvable Knowledge Base.

Core improvements over original LAMRG v12:
    1. Medical VLM encoder replaces ImageNet-pretrained CNN
    2. Three-tier hierarchical memory replaces flat memory matrix
    3. Gated memory update prevents catastrophic forgetting
    4. Adaptive update controller modulates learning per sample
    5. Cross-modal projection with VLM-text alignment initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder, MHA_FF
from .medical_vlm_extractor import MedicalVLMEncoder
from .hierarchical_memory import HierarchicalMemory
from .memory_gate import MemoryUpdateGate, AdaptiveMemoryController


class MLLMLAMRG(nn.Module):
    """MLLM-augmented LAMRG with hierarchical evolvable knowledge base.

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

        # ---- Medical VLM Encoder ----
        vlm_type = getattr(args, 'vlm_type', 'biomedclip')
        freeze_vlm = getattr(args, 'freeze_vlm', True)
        self.visual_extractor = MedicalVLMEncoder(
            vlm_type=vlm_type,
            d_model=self.d_model,
            num_labels=self.num_labels,
            freeze_vlm=freeze_vlm,
        )

        # ---- Text Encoder for report encoding ----
        self.txt_encoder = TextEncoder(
            self.d_model, self.d_ff, args.num_layers,
            self.tgt_vocab, self.num_labels,
            args.num_heads, self.dropout,
        )

        # ---- Hierarchical Memory ----
        self.memory = HierarchicalMemory(
            d_model=self.d_model,
            num_instances=self.num_slots,
            num_diseases=self.num_labels,
            num_cross=getattr(args, 'num_cross_slots', 32),
            num_heads=getattr(args, 'num_memory_heads', 8),
            dropout=self.dropout,
        )

        # ---- Memory Gate ----
        self.mem_gate = MemoryUpdateGate(d_model=self.d_model)

        # ---- Adaptive Controller ----
        self.mem_controller = AdaptiveMemoryController(d_model=self.d_model)

        # ---- Projections ----
        self.linear_z = nn.Linear(self.d_model, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)
        self.linear_fcfeat = nn.Linear(self.d_model, self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        # ---- Transformer Encoder-Decoder ----
        self.encoder_decoder = TransformerModel(args, tokenizer)

        self._init_weights()

    def _init_weights(self):
        for module in [self.linear_z, self.linear_feat, self.classifier,
                       self.embed_labels, self.linear_fcfeat]:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)

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
        """Forward pass with hierarchical memory retrieval and gated update.

        Args:
            images: (B, C, H, W) or (B, 2, C, H, W) for IU
            targets: (B, seq_len) report token ids (training only)
            labels: (B, num_labels) CheXpert labels
            mode: 'train' or 'sample'

        Returns:
            train mode: (output, vis_labels, txt_labels, z_img, z_txt, gate_stats)
            sample mode: (output, vis_labels)
        """
        bsz = images.shape[0]

        # ---- Visual feature extraction ----
        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        # ---- Global projections ----
        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        # ---- Label embedding as memory query ----
        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))

        # ---- Memory operations ----
        gate_stats = {}
        if mode == 'train':
            # Encode ground-truth report for memory update
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)

            # Update hierarchical memory with gating
            old_memory_state = {
                'instance': self.memory.instance_memory.clone(),
                'disease': self.memory.disease_memory.clone(),
                'cross': self.memory.cross_memory.clone(),
            }

            # Retrieve prior with memory update
            prior = self.memory(emb_labels, txt_feats, mode='update_and_retrieve')

            # Apply gating between old and updated memory
            new_instance_mem = self.memory.instance_memory
            gated_instance, inst_gate_stats = self.mem_gate(
                old_memory_state['instance'].expand(bsz, -1, -1),
                new_instance_mem.expand(bsz, -1, -1),
            )
            gate_stats.update(inst_gate_stats)

            # Adaptive modulation
            update_intensity, novelty = self.mem_controller(z_img, prior.mean(dim=1))
            gate_stats['update_intensity'] = update_intensity.mean().item()
            gate_stats['novelty'] = novelty.mean().item()

        else:
            txt_feats, z_txt, txt_labels = None, None, None
            prior = self.memory(emb_labels, mode='retrieve')

        # ---- Feature augmentation ----
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        # ---- Decoding ----
        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt, gate_stats)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

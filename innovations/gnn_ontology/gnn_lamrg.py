"""
GNN-LAMRG: Knowledge graph-enhanced radiology report generation.

Extends LAMRG v12 with a parallel graph neural network reasoning branch:
    1. Original path: Visual features -> Memory retrieval -> Prior knowledge
    2. GNN path: Predicted labels -> GNN reasoning over knowledge graph -> Graph knowledge
    3. Fusion: Gated combination of both knowledge sources

The GNN branch provides structured medical knowledge that complements
the data-driven patterns learned by the memory attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder, MHA_FF
from .knowledge_graph import ChestXRayKG
from .gnn_reasoner import GNNReasoner
from .gated_fusion import GatedFusion


class GNNLAMRG(nn.Module):
    """LAMRG enhanced with GNN-based structured knowledge reasoning.

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

        # ---- Visual Extractor ----
        self.visual_extractor = VisualExtractor(args)

        # ---- Text Encoder ----
        self.txt_encoder = TextEncoder(
            self.d_model, self.d_ff, args.num_layers,
            self.tgt_vocab, self.num_labels,
            args.num_heads, self.dropout,
        )

        # ---- Original Memory Path (from LAMRG v12) ----
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

        # ---- Projections ----
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)
        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        # ---- GNN Path ----
        self.knowledge_graph = ChestXRayKG(
            num_labels=self.num_labels,
            d_model=self.d_model,
            ontology_source=getattr(args, 'ontology_source', 'builtin'),
        )
        self.gnn_reasoner = GNNReasoner(
            d_model=self.d_model,
            num_labels=self.num_labels,
            num_heads=getattr(args, 'gnn_heads', 4),
            dropout=self.dropout,
        )

        # ---- Gated Fusion ----
        self.gated_fusion = GatedFusion(
            d_model=self.d_model,
            num_labels=self.num_labels,
            use_context_gate=True,
            use_label_gate=True,
        )

        # ---- Label-to-graph mapping ----
        self.label_to_graph = nn.Sequential(
            nn.Linear(self.num_labels, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )

        # ---- Transformer Encoder-Decoder ----
        self.encoder_decoder = TransformerModel(args, tokenizer)

        # ---- Loss weights ----
        self.graph_reg_weight = getattr(args, 'graph_reg_weight', 0.01)
        self.fusion_decision_weight = getattr(args, 'fusion_decision_weight', 0.01)

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

    def forward_gnn_path(self, vis_labels):
        """GNN reasoning branch: labels -> graph propagation -> reasoned knowledge.

        Args:
            vis_labels: (B, num_labels) predicted finding probabilities

        Returns:
            gnn_knowledge: (B, num_labels, d_model) per-finding GNN features
            graph_reg_loss: Scalar graph regularization loss
        """
        # Map label predictions to graph node initialization
        label_feats = self.label_to_graph(vis_labels)  # (B, d_model)

        # Run KG message passing
        kg_node_features, graph_reg_loss = self.knowledge_graph(
            node_features=self.knowledge_graph.node_embeddings,
        )

        # Combine with current label features
        combined_nodes = kg_node_features + label_feats.unsqueeze(1)

        # GNN reasoning
        reasoned_features, global_knowledge = self.gnn_reasoner(
            combined_nodes, self.knowledge_graph.edge_index,
        )

        # Expand to batch
        gnn_knowledge = reasoned_features.unsqueeze(0).expand(
            vis_labels.shape[0], -1, -1)

        return gnn_knowledge, graph_reg_loss

    def forward(self, images, targets=None, labels=None, mode='train'):
        """Forward pass with dual-path knowledge integration.

        Returns:
            train: (output, vis_labels, txt_labels, z_img, z_txt, aux_losses)
            sample: (output, vis_labels)
        """
        bsz = images.shape[0]

        # ---- Visual Feature Extraction ----
        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        # ---- Global Projections ----
        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        # ---- Path 1: Memory-based Knowledge Retrieval ----
        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        mask = self.mask.to(images).expand(bsz, -1, -1)

        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(memory, txt_feats, mask)
        else:
            txt_feats, z_txt, txt_labels = None, None, None

        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        memory_prior = self.select_prior(emb_labels, memory)

        # ---- Path 2: GNN-based Knowledge Reasoning ----
        gnn_knowledge, graph_reg_loss = self.forward_gnn_path(vis_labels)

        # ---- Gated Fusion of Both Paths ----
        fused_knowledge, gate_stats = self.gated_fusion(
            memory_prior, gnn_knowledge,
            context_features=z_img,
            label_predictions=vis_labels,
        )

        # ---- Feature Augmentation ----
        att_feats = torch.cat((att_feats, self.linear_feat(fused_knowledge)), dim=1)

        # ---- Decoding ----
        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')

            aux_losses = {
                'graph_reg_loss': graph_reg_loss,
                'gate_stats': gate_stats,
            }
            return (output, vis_labels, txt_labels, z_img, z_txt, aux_losses)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

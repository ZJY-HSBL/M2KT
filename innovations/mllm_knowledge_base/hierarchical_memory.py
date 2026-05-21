"""
Hierarchical Memory: Three-tier knowledge base inspired by the human diagnostic
reasoning process.

Tiers:
    1. Instance-level slots: Store individual case patterns (findings per image)
    2. Disease-level slots: Aggregate disease-specific knowledge across cases
    3. Cross-disease slots: Model inter-disease relationships and comorbidities

Architecture:
    [Image] -> Query -> [Instance Memory] -> [Disease Memory] -> [Cross-Disease Memory]
                                |                    |                      |
                           attention            attention              attention
                                v                    v                      v
                          instance_prior      disease_prior        cross_disease_prior
                                                      |
                                              Gated Fusion -> Final Prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HierarchicalMemory(nn.Module):
    """Three-tier hierarchical memory for multi-granularity knowledge storage.

    Args:
        d_model: Feature dimension for all memory slots
        num_instances: Number of instance-level memory slots
        num_diseases: Number of disease-level memory slots (default: num_labels)
        num_cross: Number of cross-disease association slots
        num_heads: Attention heads for memory operations
        dropout: Dropout rate
    """

    def __init__(self, d_model=512, num_instances=256, num_diseases=14,
                 num_cross=32, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_instances = num_instances
        self.num_diseases = num_diseases
        self.num_cross = num_cross

        # Three memory banks, each initialized as identity-like
        self.instance_memory = nn.Parameter(
            torch.eye(num_instances)[:, :d_model].unsqueeze(0)
            if d_model <= num_instances
            else torch.cat([
                torch.eye(num_instances),
                torch.zeros(num_instances, d_model - num_instances)
            ], dim=-1).unsqueeze(0)
        )

        self.disease_memory = nn.Parameter(
            torch.eye(num_diseases)[:, :d_model].unsqueeze(0)
            if d_model <= num_diseases
            else torch.cat([
                torch.eye(num_diseases),
                torch.zeros(num_diseases, d_model - num_diseases)
            ], dim=-1).unsqueeze(0)
        )

        self.cross_memory = nn.Parameter(torch.randn(1, num_cross, d_model) * 0.02)

        # Attention modules for each tier
        from modules.Transformer import MultiHeadedAttention, SublayerConnection

        self.instance_attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.disease_attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout)

        self.instance_update = MultiHeadedAttention(num_heads, d_model, dropout)
        self.disease_update = MultiHeadedAttention(num_heads, d_model, dropout)
        self.cross_update = MultiHeadedAttention(num_heads, d_model, dropout)

        # Projections for inter-tier communication
        self.instance_to_disease = nn.Linear(d_model, d_model)
        self.disease_to_cross = nn.Linear(d_model, d_model)

        # Output aggregation
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def update_memory(self, txt_feats, mask=None):
        """Update all three memory tiers using text features from ground-truth reports.

        Performs bottom-up update: text -> instance -> disease -> cross-disease

        Args:
            txt_feats: (B, seq_len, d_model) text encoder output features
            mask: Optional attention mask

        Returns:
            Updated (instance_memory, disease_memory, cross_memory)
        """
        bsz = txt_feats.shape[0]
        inst_mem = self.instance_memory.expand(bsz, -1, -1)
        dis_mem = self.disease_memory.expand(bsz, -1, -1)
        cross_mem = self.cross_memory.expand(bsz, -1, -1)

        # Tier 1: Update instance memory from text
        inst_mem = self.instance_update(inst_mem, txt_feats, txt_feats)

        # Tier 2: Propagate to disease memory
        inst_repr = self.instance_to_disease(inst_mem)
        dis_mem = self.disease_update(dis_mem, inst_repr, inst_repr)

        # Tier 3: Propagate to cross-disease memory
        dis_repr = self.disease_to_cross(dis_mem)
        cross_mem = self.cross_update(cross_mem, dis_repr, dis_repr)

        return inst_mem, dis_mem, cross_mem

    def retrieve_prior(self, query, inst_mem=None, dis_mem=None, cross_mem=None):
        """Retrieve prior knowledge from all three memory tiers using the query.

        Performs top-down retrieval: query -> instance -> disease -> cross-disease

        Args:
            query: (B, query_len, d_model) query features (e.g., label embeddings)
            inst_mem: Pre-updated instance memory, or None to use raw memory
            dis_mem: Pre-updated disease memory, or None
            cross_mem: Pre-updated cross-disease memory, or None

        Returns:
            fused_prior: (B, query_len, d_model) aggregated knowledge prior
        """
        bsz = query.shape[0]

        if inst_mem is None:
            inst_mem = self.instance_memory.expand(bsz, -1, -1)
        if dis_mem is None:
            dis_mem = self.disease_memory.expand(bsz, -1, -1)
        if cross_mem is None:
            cross_mem = self.cross_memory.expand(bsz, -1, -1)

        # Tier 1: Query instance memory
        inst_prior = self.instance_attn(query, inst_mem, inst_mem)

        # Tier 2: Query disease memory
        disease_prior = self.disease_attn(query, dis_mem, dis_mem)

        # Tier 3: Query cross-disease memory
        cross_prior = self.cross_attn(query, cross_mem, cross_mem)

        # Fuse all three levels of prior knowledge
        fused = torch.cat([inst_prior, disease_prior, cross_prior], dim=-1)
        fused_prior = self.fusion(fused)

        return fused_prior

    def forward(self, query, txt_feats=None, mode='retrieve'):
        """Unified forward for hierarchical memory operations.

        Args:
            query: (B, query_len, d_model) retrieval query
            txt_feats: (B, seq_len, d_model) text features for update (training only)
            mode: 'retrieve' or 'update_and_retrieve'

        Returns:
            fused_prior: Aggregated knowledge prior
        """
        inst_mem, dis_mem, cross_mem = None, None, None

        if mode == 'update_and_retrieve' and txt_feats is not None:
            inst_mem, dis_mem, cross_mem = self.update_memory(txt_feats)

        return self.retrieve_prior(query, inst_mem, dis_mem, cross_mem)

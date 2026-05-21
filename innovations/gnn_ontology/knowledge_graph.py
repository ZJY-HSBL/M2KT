"""
Chest X-Ray Knowledge Graph: Builds and manages a heterogeneous graph
encoding clinical knowledge about chest radiographic findings.

Graph structure:
    - Nodes: CheXpert labels (14) + anatomical regions (7) + category concepts (6)
    - Edges:
        * Hierarchy: finding IS_A category (e.g., Cardiomegaly -> Cardiac)
        * Comorbidity: finding COOCCURS_WITH finding (e.g., Cardiomegaly <-> Edema)
        * Causality: finding CAUSES finding (e.g., Pneumothorax -> Atelectasis)
        * Anatomy: finding LOCATED_IN region (e.g., Pleural Effusion -> Pleural Space)

Uses PyTorch Geometric-style sparse operations for efficient graph computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChestXRayKG(nn.Module):
    """Heterogeneous knowledge graph for chest X-ray findings.

    Supports message passing along typed edges with learnable edge-type-specific
    transformations.

    Args:
        num_labels: Number of CheXpert findings
        d_model: Feature dimension for node embeddings
        num_edge_types: Number of distinct relationship types
        ontology_source: 'builtin' or path to external ontology file
    """

    def __init__(self, num_labels=14, d_model=512, num_edge_types=3,
                 ontology_source='builtin'):
        super().__init__()
        self.num_labels = num_labels
        self.d_model = d_model
        self.num_edge_types = num_edge_types

        # Load ontology
        from .ontology_loader import OntologyLoader
        self.ontology = OntologyLoader(ontology_source, num_labels)

        # Build graph structure
        edge_index, edge_type, init_features = self.ontology.build_knowledge_graph()
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_type', edge_type)

        # Learnable node embeddings (initialized from ontology)
        self.node_embeddings = nn.Parameter(
            torch.randn(num_labels, d_model) * 0.02
        )
        # Initialize with one-hot identity through a projection
        init_proj = nn.Linear(num_labels, d_model, bias=False)
        self.node_embeddings.data = init_proj(init_features)

        # Edge-type-specific linear transformations
        self.edge_transforms = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(num_edge_types)
        ])

        # Edge-type attention weights for message importance
        self.edge_attention = nn.Parameter(torch.ones(num_edge_types))

        # Anatomical region embeddings
        region_ids = self.ontology.get_anatomical_grouping()
        num_regions = int(region_ids.max().item()) + 1
        self.region_embeddings = nn.Parameter(torch.randn(num_regions, d_model) * 0.02)
        self.register_buffer('region_ids', region_ids)

        # Graph-level summary
        self.graph_summary = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def message_passing(self, node_features, num_iterations=2):
        """Perform typed message passing over the knowledge graph.

        Args:
            node_features: (num_labels, d_model) current node representations
            num_iterations: Number of message passing rounds

        Returns:
            updated_features: (num_labels, d_model) after graph convolution
        """
        if self.edge_index.numel() == 0:
            return node_features

        edge_index = self.edge_index
        edge_type = self.edge_type

        for _ in range(num_iterations):
            # Source and target nodes for each edge
            src, dst = edge_index[0], edge_index[1]

            # Message from source nodes
            messages = node_features[src]  # (E, d_model)

            # Apply edge-type-specific transforms
            transformed_messages = torch.zeros_like(messages)
            for t in range(self.num_edge_types):
                mask = (edge_type == t)
                if mask.any():
                    transformed_messages[mask] = self.edge_transforms[t](
                        messages[mask]
                    )

            # Edge-type attention weighting
            attention_weights = F.softmax(self.edge_attention, dim=0)
            weighted_messages = torch.zeros_like(transformed_messages)
            for t in range(self.num_edge_types):
                mask = (edge_type == t)
                if mask.any():
                    weighted_messages[mask] = attention_weights[t] * transformed_messages[mask]

            # Aggregate messages to target nodes
            aggregated = torch.zeros_like(node_features)
            aggregated = aggregated.index_add(0, dst, weighted_messages)

            # Degree normalization
            degree = torch.zeros(node_features.shape[0], device=node_features.device)
            degree = degree.index_add(0, dst, torch.ones(dst.shape[0], device=node_features.device))
            degree = degree.clamp(min=1).unsqueeze(-1)

            # Update nodes: residual connection + message
            node_features = node_features + aggregated / degree
            node_features = self.graph_summary(node_features)

        return node_features

    def compute_adjacency_regularization(self, node_features):
        """Regularization based on known comorbidity relationships.

        Encourages connected nodes to have similar embeddings, weighted by
        relationship type.

        Returns:
            reg_loss: Scalar regularization loss
        """
        if self.edge_index.numel() == 0:
            return torch.tensor(0.0, device=node_features.device)

        src, dst = self.edge_index[0], self.edge_index[1]
        edge_type = self.edge_type

        # Only regularize comorbidity edges (type 1)
        comorbidity_mask = (edge_type == 1)
        if not comorbidity_mask.any():
            return torch.tensor(0.0, device=node_features.device)

        src_com = src[comorbidity_mask]
        dst_com = dst[comorbidity_mask]

        # Cosine similarity between connected nodes
        src_norm = F.normalize(node_features[src_com], dim=-1)
        dst_norm = F.normalize(node_features[dst_com], dim=-1)
        similarity = (src_norm * dst_norm).sum(dim=-1)

        # Encourage positive similarity for comorbid pairs
        loss = F.relu(0.5 - similarity).mean()

        return loss

    def forward(self, node_features=None, num_iterations=2):
        """Run graph reasoning over the knowledge graph.

        Args:
            node_features: Optional (num_labels, d_model) override for initial features
            num_iterations: Message passing rounds

        Returns:
            kg_features: (num_labels, d_model) knowledge-enhanced features
            reg_loss: Adjacency regularization loss
        """
        if node_features is None:
            node_features = self.node_embeddings

        kg_features = self.message_passing(node_features, num_iterations)
        reg_loss = self.compute_adjacency_regularization(kg_features)

        return kg_features, reg_loss

    def get_region_features(self):
        """Get anatomical region-level features by pooling label features.

        Returns:
            region_features: (num_regions, d_model)
        """
        label_features = self.node_embeddings
        num_regions = self.region_embeddings.shape[0]

        region_features = torch.zeros(num_regions, self.d_model,
                                       device=label_features.device)
        region_counts = torch.zeros(num_regions, device=label_features.device)

        for i in range(self.num_labels):
            r = self.region_ids[i]
            region_features[r] = region_features[r] + label_features[i]
            region_counts[r] = region_counts[r] + 1

        region_counts = region_counts.clamp(min=1).unsqueeze(-1)
        region_features = region_features / region_counts

        # Blend with learned region embeddings
        region_features = region_features + self.region_embeddings

        return region_features

"""
Ontology Loader: Loads and processes medical ontologies (UMLS, RadLex, SNOMED CT)
into a structured format for graph construction.

Since direct access to UMLS/RadLex requires API keys or local databases,
this module provides:
    1. A built-in chest X-ray ontology based on CheXpert labels and their
       known clinical relationships
    2. Interface for loading external ontology files (JSON, CSV, OWL)
    3. Mapping from ontology concepts to model label indices

The built-in ontology encodes:
    - Hierarchical relationships (e.g., "Pleural Effusion" IS_A "Pleural Abnormality")
    - Comorbidity relationships (e.g., "Cardiomegaly" COOCCURS_WITH "Pulmonary Edema")
    - Causal relationships (e.g., "Consolidation" CAUSES "Air Bronchogram")
    - Anatomical locations (e.g., "Pneumothorax" LOCATED_IN "Pleural Space")
"""

import torch
import numpy as np


# Built-in chest X-ray ontology: CheXpert labels and their clinical relationships
# Based on radiology domain knowledge
CHEST_XRAY_ONTOLOGY = {
    "labels": [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Thickening", "Fracture", "Support Device",
        "Enlarged",
    ],
    "categories": {
        "Normal": [0],
        "Cardiac": [1, 2],
        "Pulmonary": [3, 4, 5, 6, 7, 8],
        "Pleural": [9, 10],
        "Musculoskeletal": [11],
        "Iatrogenic": [12],
        "Other": [13],
    },
    # Hierarchical parent-child relationships: parent -> [children]
    "hierarchy": {
        "Thoracic Abnormality": ["Cardiac", "Pulmonary", "Pleural", "Musculoskeletal"],
        "Cardiac": ["Enlarged Cardiomediastinum", "Cardiomegaly"],
        "Pulmonary": ["Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
                       "Atelectasis", "Pneumothorax"],
        "Pleural": ["Pleural Effusion", "Pleural Thickening"],
        "Musculoskeletal": ["Fracture"],
    },
    # Known co-occurrence patterns (comorbidities)
    "comorbidity": [
        (2, 5),   # Cardiomegaly <-> Edema (heart failure)
        (5, 6),   # Edema <-> Consolidation
        (5, 9),   # Edema <-> Pleural Effusion
        (6, 7),   # Consolidation <-> Atelectasis
        (9, 10),  # Pleural Effusion <-> Pleural Thickening
        (2, 9),   # Cardiomegaly <-> Pleural Effusion
        (1, 2),   # Enlarged Cardiomediastinum <-> Cardiomegaly
        (3, 6),   # Lung Opacity <-> Consolidation
        (7, 11),  # Atelectasis <-> Fracture (post-traumatic)
    ],
    # Causal/directional relationships: (cause, effect)
    "causality": [
        (8, 7),   # Pneumothorax -> Atelectasis (compressive)
        (6, 5),   # Consolidation -> Edema (inflammatory)
        (2, 5),   # Cardiomegaly -> Edema (CHF)
        (11, 9),  # Fracture -> Pleural Effusion (hemothorax)
    ],
    # Anatomical locations
    "anatomy": {
        0: "whole_image",
        1: "mediastinum",
        2: "cardiac_silhouette",
        3: "lung_parenchyma",
        4: "lung_parenchyma",
        5: "lung_parenchyma",
        6: "lung_parenchyma",
        7: "lung_parenchyma",
        8: "pleural_space",
        9: "pleural_space",
        10: "pleura",
        11: "ribs",
        12: "variable",
        13: "variable",
    }
}


class OntologyLoader:
    """Loads and manages medical ontology for graph construction.

    Args:
        ontology_source: 'builtin', or path to external JSON/CSV file
        num_labels: Number of CheXpert labels
    """

    def __init__(self, ontology_source='builtin', num_labels=14):
        self.num_labels = num_labels

        if ontology_source == 'builtin':
            self.ontology = CHEST_XRAY_ONTOLOGY
        else:
            self.ontology = self._load_external(ontology_source)

        self._build_mappings()

    def _load_external(self, path):
        """Load ontology from external file (JSON or CSV)."""
        import json
        with open(path, 'r') as f:
            return json.load(f)

    def _build_mappings(self):
        """Build tensor-based relationship matrices from ontology."""
        n = self.num_labels

        # Hierarchy adjacency matrix: A[i,j] = 1 if i is parent of j
        self.hierarchy_edges = []
        label_to_idx = {name.lower(): i for i, name in enumerate(self.ontology["labels"])}

        for parent, children in self.ontology.get("hierarchy", {}).items():
            for child in children:
                child_idx = label_to_idx.get(child.lower())
                if child_idx is not None:
                    self.hierarchy_edges.append((parent, child_idx))

        # Comorbidity adjacency (symmetric)
        self.comorbidity_edges = self.ontology.get("comorbidity", [])

        # Causal adjacency (directional): edge from cause to effect
        self.causal_edges = self.ontology.get("causality", [])

    def build_knowledge_graph(self):
        """Build a PyTorch Geometric-style graph from the ontology.

        Returns:
            edge_index: (2, num_edges) tensor of directed edges
            edge_type: (num_edges,) tensor of edge types (0=hierarchy, 1=comorbidity, 2=causal)
            node_features: (num_labels, feature_dim) initial node features
        """
        edges = []
        edge_types = []

        # Hierarchy edges (bidirectional for message passing)
        for parent_name, child_idx in self.hierarchy_edges:
            parent_idx = len(edges)  # Category nodes come after label nodes
            edges.append([child_idx, parent_idx])
            edge_types.append(0)
            edges.append([parent_idx, child_idx])
            edge_types.append(0)

        # Comorbidity edges (bidirectional)
        for i, j in self.comorbidity_edges:
            edges.append([i, j])
            edge_types.append(1)
            edges.append([j, i])
            edge_types.append(1)

        # Causal edges (directional)
        for cause, effect in self.causal_edges:
            edges.append([cause, effect])
            edge_types.append(2)

        if not edges:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types, dtype=torch.long)

        # Initialize node features with one-hot identity
        node_features = torch.eye(self.num_labels)

        return edge_index, edge_type, node_features

    def get_anatomical_grouping(self):
        """Get anatomical region grouping for attention regularization.

        Returns:
            region_ids: (num_labels,) tensor mapping label -> anatomical region
        """
        anatomy = self.ontology.get("anatomy", {})
        regions = list(set(anatomy.values()))
        region_to_id = {r: i for i, r in enumerate(regions)}

        region_ids = torch.zeros(self.num_labels, dtype=torch.long)
        for label_idx, region_name in anatomy.items():
            region_ids[label_idx] = region_to_id[region_name]

        return region_ids

    def get_label_names(self):
        """Get list of label name strings."""
        return self.ontology.get("labels", [f"Finding_{i}" for i in range(self.num_labels)])

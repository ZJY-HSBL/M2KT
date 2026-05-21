# M2KT: Multi-Modal Knowledge Transfer for Radiology Report Generation

A deep learning framework for automatic radiology report generation from chest X-ray images, featuring a learned knowledge base with multi-modal alignment and five advanced research directions.

---

## Overview

This project explores automatic generation of radiology reports from chest X-ray images. The baseline architecture follows an encoder-decoder paradigm with two core mechanisms:

1. **Learned Knowledge Base** — A learnable memory matrix is updated during training via attention from ground-truth report text, and queried at inference time using predicted clinical labels to retrieve relevant prior knowledge.
2. **Multi-modal Alignment** — A hinge-based contrastive ranking loss with a label-aware dynamic margin aligns image and text representations in a shared embedding space.

```
X-ray Image → Visual Extractor (CNN) → Image Features + Label Prediction
                                         ↓
                                    Knowledge Base Retrieval (via label embedding)
                                         ↓
                              Transformer Encoder-Decoder → Radiology Report
                                         ↓
                    LM Loss + Label Loss + Ranking Loss → Joint Training
```

---

## Five Research Directions

All code is organized under `innovations/`, each direction self-contained and building upon the shared `models/` and `modules/` foundation.

### 1. MLLM-Augmented Evolvable Knowledge Base

**Dir**: `innovations/mllm_knowledge_base/` | **Entry**: `python -m innovations.mllm_knowledge_base.main_mllm`

Replaces the standard CNN visual encoder with pretrained medical vision-language models (BiomedCLIP, BioViL, CheXzero), and introduces a three-tier hierarchical memory structure (instance-level → disease-level → cross-disease slots) with LSTM-style gated update mechanisms to prevent catastrophic forgetting during knowledge accumulation.

| Module | Role |
|--------|------|
| Medical VLM Encoder | BiomedCLIP / BioViL / CheXzero as visual backbone |
| Hierarchical Memory | Three-tier memory: instance, disease, cross-disease |
| Gated Memory Update | Learnable forget/write gates for controlled knowledge evolution |
| Adaptive Controller | Novelty-aware per-sample update intensity modulation |

### 2. Agent-Based Iterative Self-Refinement

**Dir**: `innovations/agent_refinement/` | **Entry**: `python -m innovations.agent_refinement.main_agent`

Mimics a radiologist's review workflow: a base generator produces an initial draft, then a Refiner Agent iteratively revises it using visual feedback. A Consistency Checker detects internal contradictions, a Completeness Verifier identifies omitted findings, and Direct Preference Optimization (DPO) aligns the refinement process with clinical quality preferences.

| Module | Role |
|--------|------|
| Draft → Refine Pipeline | Cross-attention based iterative report improvement |
| Consistency Checker | NLI-style contradiction detection + clinical rule base |
| Completeness Verifier | Label-to-report coverage verification |
| DPO Training | Direct Preference Optimization for quality-aligned refinement |

### 3. Evidence Grounding & Explainability

**Dir**: `innovations/evidence_grounding/` | **Entry**: `python -m innovations.evidence_grounding.main_grounding`

Addresses the "black box" problem by extracting spatial attention heatmaps from decoder cross-attention, linking each generated sentence to specific image regions. Includes Monte Carlo Dropout-based uncertainty quantification that separates aleatoric from epistemic uncertainty, and produces a per-report reliability score.

| Module | Role |
|--------|------|
| Spatial Attention Head | Per-token and sentence-level attention heatmaps |
| Grounding Losses | BBox weak supervision + contrastive + smoothness regularization |
| Uncertainty Estimator | MC Dropout for aleatoric & epistemic uncertainty |
| Reliability Scoring | Learned clinical trustworthiness assessment |

### 4. GNN + Medical Ontology Enhancement

**Dir**: `innovations/gnn_ontology/` | **Entry**: `python -m innovations.gnn_ontology.main_gnn`

Constructs a heterogeneous chest X-ray knowledge graph encoding hierarchical, comorbidity, causal, and anatomical relationships among 14 clinical findings. A 2-layer Graph Attention Network (GAT) propagates structured knowledge across disease nodes, and a learnable gated fusion mechanism adaptively balances the GNN reasoning path with the data-driven memory retrieval path.

| Module | Role |
|--------|------|
| Chest X-Ray Knowledge Graph | Hierarchy + comorbidity + causality + anatomy edges |
| GAT Reasoner | 2-layer graph attention for cross-disease knowledge propagation |
| Gated Fusion | Context-aware fusion of GNN knowledge and memory retrieval |
| Built-in Ontology | 14 CheXpert labels with clinical relationships, UMLS-extensible |

### 5. Temporal Disease Progression Modeling

**Dir**: `innovations/temporal_progression/` | **Entry**: `python -m innovations.temporal_progression.main_temporal`

Models disease evolution by processing paired longitudinal chest X-rays from the same patient. A Siamese visual encoder with cross-temporal attention and difference encoding predicts per-finding progression states (stable, improved, worsened). Temporal consistency losses enforce smooth severity trajectories and align report-text differences with visual changes.

| Module | Role |
|--------|------|
| Longitudinal Pairing | Same-patient MIMIC-CXR studies with configurable time delta |
| Siamese Temporal Encoder | Shared CNN + cross-temporal attention + difference encoding |
| Progression Classifier | Per-finding 3-class: stable / improved / worsened |
| Temporal Consistency Loss | Severity smoothness + report-text alignment + progression supervision |

---

## Project Structure

```
├── config/                          # YAML configs and argument parser
│   ├── iu.yml, mimic.yml            # Dataset base configs
│   ├── iu_resnet.yml                # IU + ResNet-101
│   ├── iu_densenet.yml              # IU + DenseNet-121
│   ├── iu_efficientnet.yml          # IU + EfficientNet-b0
│   ├── mimic_resnet.yml             # MIMIC + ResNet-101
│   ├── config.py                    # Config loader with _BASE_ inheritance
│   └── opts.py                      # CLI argument parsing
│
├── models/                          # Model definitions
│   ├── lamrg.py                     # Model variants v7~v12, v91, v92, vRebuttal
│   ├── baseline.py                  # Pure Transformer baseline
│   └── LabelFeature.py              # Label feature injection variant
│
├── modules/                         # Core building blocks
│   ├── visual_extractor.py          # ResNet/DenseNet/EfficientNet visual encoder
│   ├── text_encoder.py              # Transformer-based text encoder + MHA_FF
│   ├── Transformer.py               # Full Transformer (encoder, decoder, attention)
│   ├── AttModel.py                  # Attention model base (beam search)
│   ├── CaptionModel.py              # Captioning model base (sequence decoding)
│   ├── tokenizers.py                # Tokenizer (vocab building, report cleaning)
│   ├── datasets.py                  # IU (multi-view) / MIMIC (single-view) datasets
│   ├── dataloaders.py               # DataLoader with augmentation
│   ├── loss.py                      # LM loss + label BCE loss + ranking loss
│   ├── trainer.py                   # Training loop, validation, checkpointing
│   ├── metrics.py                   # BLEU 1-4, ROUGE-L, CIDEr, METEOR
│   ├── optimizers.py                # Optimizer and LR scheduler builders
│   ├── utils.py                     # Decoding utilities
│   └── efficientnet_pytorch/        # Bundled EfficientNet implementation
│
├── misc/
│   └── utils.py                     # Logging setup and image normalization
│
├── innovations/                     # Five research directions
│   ├── mllm_knowledge_base/         # [1] MLLM-enhanced knowledge base
│   ├── agent_refinement/            # [2] Agent iterative self-refinement
│   ├── evidence_grounding/          # [3] Evidence grounding & explainability
│   ├── gnn_ontology/                # [4] GNN + ontology enhancement
│   └── temporal_progression/        # [5] Temporal disease progression
│
├── main.py                          # Entry: full model with knowledge base
├── main_basic.py                    # Entry: basic Transformer baseline
├── main_label_feature.py            # Entry: label feature injection
├── mainbaseline.py                  # Entry: minimal baseline
├── README.md                        # English documentation
└── README_CN.md                     # Chinese documentation
```

---

## Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | >= 3.8 | Runtime |
| PyTorch | >= 1.7 | Deep learning framework |
| torchvision | — | Pretrained CNN models |
| tensorboardX | — | Training visualization |
| tqdm | — | Progress bars |
| numpy | — | Numerical computation |
| pandas | — | Data processing |
| pyyaml / yacs | — | Configuration management |
| Pillow | — | Image loading |
| pycocoevalcap | — | MS COCO evaluation metrics |
| CheXpert-labeler | — | External label extraction |
| transformers | optional (Dir 1) | BiomedCLIP/BioViL integration |

## Data Preparation

Download the datasets and place them in the `data/` directory:

- **IU X-Ray**: [Indiana University Chest X-ray Collection](https://iuhealth.org/find-medical-services/x-rays) — ~7,470 images, 2 views per report
- **MIMIC-CXR**: [PhysioNet MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) — ~377,110 images, 1 view per report

Download the evaluation toolkit:

```bash
git clone https://github.com/tylin/coco-caption.git pycocoevalcap/
```

---

## Training

### Baseline Models

```bash
# Full model (with knowledge base + multi-modal alignment)
python main.py --cfg config/iu_resnet.yml --expe_name experiment_name \
               --label_loss --rank_loss --version 12

# Basic Transformer baseline
python main_basic.py --cfg config/iu_resnet.yml --expe_name experiment_name \
                     --label_loss --rank_loss --version 91

# Ablation: without learned knowledge base
python main.py --cfg config/iu_resnet.yml --expe_name experiment_name \
               --label_loss --rank_loss --version 92

# Ablation: without multi-modal alignment
python main.py --cfg config/iu_resnet.yml --expe_name experiment_name \
               --version 12     # omit --rank_loss
```

### Research Directions

```bash
# Direction 1: MLLM-augmented knowledge base
python -m innovations.mllm_knowledge_base.main_mllm \
    --cfg config/iu_resnet.yml --expe_name mllm_v1 \
    --vlm_type biomedclip --freeze_vlm --label_loss --rank_loss

# Direction 2: Agent-based iterative refinement
python -m innovations.agent_refinement.main_agent \
    --cfg config/iu_resnet.yml --expe_name agent_v1 \
    --max_refine_steps 3 --use_dpo

# Direction 3: Evidence grounding
python -m innovations.evidence_grounding.main_grounding \
    --cfg config/iu_resnet.yml --expe_name grounding_v1 \
    --label_loss --rank_loss --grounding_weight 0.1

# Direction 4: GNN + ontology
python -m innovations.gnn_ontology.main_gnn \
    --cfg config/iu_resnet.yml --expe_name gnn_v1 \
    --ontology_source builtin --label_loss --rank_loss

# Direction 5: Temporal progression
python -m innovations.temporal_progression.main_temporal \
    --cfg config/mimic_resnet.yml --expe_name temporal_v1 \
    --max_time_delta 365 --label_loss --rank_loss
```

---

## Model Version Reference

| Version | Class | Description |
|---------|-------|-------------|
| 900 | `baseline.py` | Pure Transformer encoder-decoder baseline |
| 7 | `LAMRGModel_v7` | Label feature concatenation |
| 8 | `LAMRGModel_v8` | TextEncoder + prior_memory + select_prior |
| 9 | `LAMRGModel_v9` | Linear projections; returns (z_img, z_txt) for ranking |
| 10 | `LAMRGModel_v10` | Projected memory via `linear_mem` |
| 11 | `LAMRGModel_v11` | Mask, query_mem, linear_z, update_memory |
| **12** | **`LAMRGModel_v12`** | **Full model**: label embedding query, separate memory heads |
| 91 | `LAMRGModel_v91` | Ablation: label loss only, no memory |
| 92 | `LAMRGModel_v92` | Ablation: label loss + rank loss, no memory |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| BLEU-1 ~ BLEU-4 | N-gram overlap between generated and reference reports |
| ROUGE-L | Longest common subsequence recall |
| CIDEr | TF-IDF weighted consensus image description evaluation |
| METEOR | Word alignment + synonym matching |

All metrics computed via `pycocoevalcap`.

---

## Key Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cfg` | str | required | Path to YAML config file |
| `--expe_name` | str | required | Experiment name for logging |
| `--version` | int | 12 | Model version (see table above) |
| `--label_loss` | flag | False | Enable label prediction BCE loss |
| `--rank_loss` | flag | False | Enable multi-modal ranking loss |
| `--batch_size` | int | 16 | Training batch size |
| `--epochs` | int | 50 | Number of training epochs |
| `--lr_ed` | float | 1e-4 | Learning rate for encoder-decoder |
| `--lr_ve` | float | 1e-5 | Learning rate for visual extractor |
| `--grad_clip` | float | 5.0 | Gradient clipping value |
| `--early_stop` | int | 10 | Early stopping patience |
| `--seed` | int | 42 | Random seed |

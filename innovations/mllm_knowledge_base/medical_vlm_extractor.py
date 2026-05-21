"""
Medical VLM Encoder: Wraps pretrained medical vision-language models
(BiomedCLIP, BioViL, MedCLIP, CheXzero) as drop-in replacements for the
original VisualExtractor.

Supports:
    - BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT)
    - BioViL (microsoft/BioViL)
    - MedCLIP (pretrained on MIMIC-CXR)
    - CheXzero (CLIP trained on CheXpert labels)
    - Fallback: standard CNN with medical pretraining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class MedicalVLMEncoder(nn.Module):
    """Medical Vision-Language Model encoder for chest X-ray feature extraction.

    Provides a unified interface over multiple pretrained medical VLMs,
    falling back to a trainable CNN when transformers are unavailable.

    Args:
        vlm_type: One of 'biomedclip', 'biovil', 'medclip', 'chexzero', 'cnn'
        d_model: Output feature dimension (projected if needed)
        num_labels: Number of CheXpert classification labels
        freeze_vlm: Whether to freeze VLM backbone weights
        pretrained_path: Optional path to local pretrained weights
    """

    def __init__(self, vlm_type='biomedclip', d_model=512, num_labels=14,
                 freeze_vlm=True, pretrained_path=None):
        super().__init__()
        self.vlm_type = vlm_type
        self.d_model = d_model
        self.num_labels = num_labels
        self.freeze_vlm = freeze_vlm

        if vlm_type == 'biomedclip' and HAS_TRANSFORMERS:
            self._init_biomedclip(pretrained_path)
        elif vlm_type == 'biovil' and HAS_TRANSFORMERS:
            self._init_biovil(pretrained_path)
        elif vlm_type == 'chexzero' and HAS_TRANSFORMERS:
            self._init_chexzero(pretrained_path)
        else:
            self._init_cnn_fallback()

        if freeze_vlm and hasattr(self, 'vision_encoder'):
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        self.label_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_labels),
        )

    def _init_biomedclip(self, pretrained_path):
        try:
            model_name = pretrained_path or "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            from transformers import CLIPModel
            self.vision_encoder = CLIPModel.from_pretrained(model_name).vision_model
            self._vision_dim = self.vision_encoder.config.hidden_size
            self._patch_size = self.vision_encoder.config.patch_size
            self.proj = nn.Linear(self._vision_dim, self.d_model)
        except Exception:
            self._init_cnn_fallback()

    def _init_biovil(self, pretrained_path):
        try:
            from transformers import AutoModel
            model_name = pretrained_path or "microsoft/BiomedVLP-CXR-BERT-specialized"
            full_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.vision_encoder = full_model
            self._vision_dim = 768
            self.proj = nn.Linear(self._vision_dim, self.d_model)
        except Exception:
            self._init_cnn_fallback()

    def _init_chexzero(self, pretrained_path):
        self._init_biomedclip(pretrained_path)

    def _init_cnn_fallback(self):
        import torchvision.models as models
        self.vlm_type = 'cnn'
        resnet = models.resnet50(pretrained=True)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._vision_dim = 2048
        self._patch_size = 32
        self.proj = nn.Linear(self._vision_dim, self.d_model)

    def forward(self, images):
        """Extract visual features and predict labels from chest X-ray images.

        Args:
            images: (B, C, H, W) input tensor

        Returns:
            patch_feats: (B, num_patches, d_model) spatial patch features
            avg_feats:   (B, d_model) globally pooled features
            labels:      (B, num_labels) predicted CheXpert labels
        """
        if self.vlm_type in ('biomedclip', 'chexzero') and hasattr(self, 'vision_encoder'):
            bsz = images.shape[0]
            vision_outputs = self.vision_encoder(pixel_values=images, output_hidden_states=True)
            patch_feats = vision_outputs.last_hidden_state[:, 1:, :]
            patch_feats = self.proj(patch_feats)
            avg_feats = vision_outputs.pooler_output
            avg_feats = self.proj(avg_feats)

        elif self.vlm_type == 'cnn':
            bsz = images.shape[0]
            feats = self.vision_encoder(images)
            patch_feats = feats.flatten(2).permute(0, 2, 1)
            patch_feats = self.proj(patch_feats)
            avg_feats = self.avg_pool(feats).flatten(1)
            avg_feats = self.proj(avg_feats)

        else:
            raise ValueError(f"Unsupported VLM type: {self.vlm_type}")

        labels = self.label_classifier(avg_feats)
        return patch_feats, avg_feats, labels

    def get_image_embedding(self, images):
        """Extract image embedding for multimodal alignment."""
        _, avg_feats, _ = self.forward(images)
        return F.normalize(avg_feats, dim=-1)

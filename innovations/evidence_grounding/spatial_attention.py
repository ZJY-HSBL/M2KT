"""
Spatial Attention Head: Extracts interpretable spatial heatmaps from the
Transformer decoder's cross-attention to visual features.

For each generated token, the cross-attention weights over image patches
produce a spatial attention map, showing which image regions influenced
that token. When aggregated to the sentence level, this provides
sentence-to-region grounding.

Key features:
    - Token-level spatial attention maps
    - Sentence-level aggregation with adaptive pooling
    - Multi-head attention fusion (different heads attend to different regions)
    - Gradient-based attribution (integrated gradients for fine-grained maps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialAttentionHead(nn.Module):
    """Extract and aggregate spatial attention from decoder cross-attention.

    Args:
        d_model: Feature dimension
        num_heads: Number of attention heads
        image_size: Input image spatial size (H, W)
        patch_size: Vision encoder patch size
    """

    def __init__(self, d_model=512, num_heads=8, image_size=(224, 224), patch_size=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.image_size = image_size
        self.patch_size = patch_size

        # Compute grid dimensions
        self.grid_h = image_size[0] // patch_size
        self.grid_w = image_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # Learnable head importance weights for fusing multi-head attention
        self.head_importance = nn.Parameter(torch.ones(num_heads) / num_heads)

        # Adaptive pooling for sentence-level aggregation
        self.sentence_pool = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Finding-type classifier on attention maps
        self.finding_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.grid_h * self.grid_w, 128),
            nn.GELU(),
            nn.Linear(128, 14),  # 14 CheXpert findings
        )

    def extract_attention_maps(self, cross_attn_weights, head_indices=None):
        """Extract spatial attention maps from cross-attention weights.

        Args:
            cross_attn_weights: (B, num_heads, tgt_len, src_len) attention weights
                                from decoder cross-attention layer
            head_indices: Optional subset of heads to use

        Returns:
            spatial_maps: (B, tgt_len, grid_h, grid_w) spatial attention maps
        """
        bsz, num_heads, tgt_len, src_len = cross_attn_weights.shape

        # Separate CLS token from patch tokens
        num_patches = src_len - 1  # First token is usually CLS
        patch_attn = cross_attn_weights[:, :, :, 1:num_patches + 1]  # (B, H, T, P)

        if head_indices is not None:
            patch_attn = patch_attn[:, head_indices]

        # Weighted fusion across heads
        head_weights = F.softmax(self.head_importance, dim=0)
        head_weights = head_weights.view(1, -1, 1, 1)
        fused_attn = (patch_attn * head_weights).sum(dim=1)  # (B, T, P)

        # Reshape to spatial grid
        grid_h = int(np.sqrt(num_patches))
        grid_w = grid_h
        if grid_h * grid_w != num_patches:
            grid_h = int(np.sqrt(num_patches))
            grid_w = num_patches // grid_h
            if grid_h * grid_w < num_patches:
                grid_w += 1
                pad_size = grid_h * grid_w - num_patches
                fused_attn = F.pad(fused_attn, (0, pad_size))

        spatial_maps = fused_attn.view(bsz, tgt_len, grid_h, grid_w)

        # Upsample to full image resolution
        spatial_maps = F.interpolate(
            spatial_maps.view(bsz * tgt_len, 1, grid_h, grid_w),
            size=self.image_size,
            mode='bilinear',
            align_corners=False,
        ).view(bsz, tgt_len, self.image_size[0], self.image_size[1])

        return spatial_maps

    def aggregate_to_sentence(self, spatial_maps, token_to_sentence_idx):
        """Aggregate token-level maps to sentence-level.

        Args:
            spatial_maps: (B, tgt_len, H, W) per-token spatial maps
            token_to_sentence_idx: (tgt_len,) index mapping token -> sentence

        Returns:
            sentence_maps: (B, num_sentences, H, W) sentence-level maps
        """
        bsz, tgt_len, h, w = spatial_maps.shape
        num_sentences = int(token_to_sentence_idx.max().item()) + 1

        sentence_maps = torch.zeros(bsz, num_sentences, h, w,
                                     device=spatial_maps.device)

        for s in range(num_sentences):
            mask = (token_to_sentence_idx == s).float()
            mask = mask.view(1, tgt_len, 1, 1)
            sentence_maps[:, s] = (spatial_maps * mask).sum(dim=1) / mask.sum().clamp(min=1)

        return sentence_maps

    def compute_saliency(self, model, images, target_tokens, class_idx=None):
        """Compute gradient-based saliency maps (Integrated Gradients style).

        Args:
            model: The full model (for gradient computation)
            images: (B, C, H, W) input images
            target_tokens: (B, seq_len) target token indices
            class_idx: Optional specific token index to explain

        Returns:
            saliency_maps: (B, H, W) gradient-based saliency maps
        """
        images = images.clone().detach().requires_grad_(True)
        output = model(images, mode='sample')

        if class_idx is not None:
            score = output[0][:, class_idx].sum()
        else:
            score = output[0].max(dim=-1)[0].sum()

        grad = torch.autograd.grad(score, images,
                                    create_graph=False,
                                    retain_graph=False)[0]

        saliency = grad.abs().max(dim=1)[0]  # Max over channels
        saliency = saliency / (saliency.max() + 1e-8)

        return saliency

    def forward(self, cross_attn_weights, token_to_sentence_idx=None):
        """Extract and optionally aggregate spatial attention maps.

        Args:
            cross_attn_weights: (B, num_heads, tgt_len, src_len)
            token_to_sentence_idx: Optional sentence mapping

        Returns:
            spatial_maps: (B, tgt_len, H, W) per-token maps
            sentence_maps: (B, num_sent, H, W) or None
            finding_scores: (B, 14) finding predictions from attention
        """
        spatial_maps = self.extract_attention_maps(cross_attn_weights)

        sentence_maps = None
        if token_to_sentence_idx is not None:
            sentence_maps = self.aggregate_to_sentence(spatial_maps, token_to_sentence_idx)

        # Predict findings from attention distribution
        finding_input = spatial_maps.mean(dim=1).unsqueeze(1)  # (B, 1, H, W)
        finding_feats = self.sentence_pool(finding_input)
        finding_scores = self.finding_classifier(finding_feats)

        return spatial_maps, sentence_maps, finding_scores

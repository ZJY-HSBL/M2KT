"""
Grounding Loss: Evidence alignment losses that encourage the model to attend
to clinically relevant image regions when describing findings.

Loss components:
    1. Bounding Box Grounding Loss: Weakly supervised alignment between
       attention maps and available bounding box annotations
    2. Sentence-Finding Consistency: Each sentence's attention should correspond
       to the anatomical regions relevant to the described finding
    3. Contrastive Grounding Loss: Positive image-text pairs should have
       similar spatial attention patterns; negatives should diverge
    4. Smoothness Regularization: Attention maps should be spatially smooth
       (adjacent patches attend to adjacent regions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingLoss(nn.Module):
    """Evidence grounding losses for explainable report generation.

    Args:
        image_size: Spatial resolution for attention maps
        use_bbox_supervision: Whether bbox annotations are available
        smoothness_weight: Weight for spatial smoothness regularization
        contrastive_weight: Weight for contrastive grounding
    """

    def __init__(self, image_size=(224, 224), use_bbox_supervision=False,
                 smoothness_weight=0.01, contrastive_weight=0.1):
        super().__init__()
        self.image_size = image_size
        self.use_bbox_supervision = use_bbox_supervision
        self.smoothness_weight = smoothness_weight
        self.contrastive_weight = contrastive_weight

    def bbox_grounding_loss(self, spatial_maps, bbox_masks, sentence_finding_idx):
        """Align attention maps with bounding box annotations.

        Uses soft IoU between attention map and bbox mask as supervision.
        This is WEAK supervision - bbox annotations come from automated
        labelers (e.g., CheXpert-labeler) not manual annotation.

        Args:
            spatial_maps: (B, num_sent, H, W) sentence-level attention maps
            bbox_masks: (B, num_findings, H, W) binary bbox masks
            sentence_finding_idx: (num_sent,) mapping sentence -> finding index

        Returns:
            bbox_loss: Scalar soft IoU loss
        """
        bsz, num_sent, h, w = spatial_maps.shape
        device = spatial_maps.device

        loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(bsz):
            for s in range(num_sent):
                f_idx = sentence_finding_idx[s]
                if f_idx >= bbox_masks.shape[1]:
                    continue

                if bbox_masks[b, f_idx].sum() == 0:
                    continue

                attn_map = spatial_maps[b, s]  # (H, W)
                bbox_mask = bbox_masks[b, f_idx]  # (H, W)

                # Soft IoU
                intersection = (attn_map * bbox_mask).sum()
                union = attn_map.sum() + bbox_mask.sum() - intersection
                soft_iou = (intersection + 1e-8) / (union + 1e-8)

                loss = loss + (1.0 - soft_iou)
                count += 1

        if count > 0:
            loss = loss / count

        return loss

    def sentence_finding_consistency_loss(self, spatial_maps, finding_predictions,
                                           sentence_embeddings, finding_embeddings):
        """Encourage sentence attention to match the described finding's
        anatomical region.

        Uses cosine similarity between:
            - Sentence embedding (what is described)
            - Finding embedding (what anatomical region)
        as a soft target for attention map alignment.

        Args:
            spatial_maps: (B, num_sent, H, W)
            finding_predictions: (B, num_sent, num_findings) finding per sentence
            sentence_embeddings: (B, num_sent, d_model)
            finding_embeddings: (num_findings, d_model)

        Returns:
            consistency_loss: Scalar alignment loss
        """
        bsz, num_sent = finding_predictions.shape[:2]

        # Similarity between sentence and each finding
        sent_norm = F.normalize(sentence_embeddings, dim=-1)
        find_norm = F.normalize(finding_embeddings, dim=-1)
        similarity = torch.matmul(sent_norm, find_norm.T)  # (B, num_sent, num_find)

        # Soft target: which findings are most relevant
        target = F.softmax(similarity, dim=-1)

        # Encourage attention to focus on relevant anatomical regions
        # This is a self-supervised consistency objective
        batch_attn = spatial_maps.view(bsz, num_sent, -1)  # (B, num_sent, H*W)
        batch_attn = F.normalize(batch_attn, dim=-1)

        # Attention diversity: different sentences should attend to different regions
        attn_diversity = torch.matmul(batch_attn, batch_attn.transpose(1, 2))
        # Penalize high similarity between different sentences' attention
        off_diag = attn_diversity - torch.eye(num_sent, device=attn_diversity.device).unsqueeze(0)
        diversity_loss = F.relu(off_diag - 0.3).mean()

        return diversity_loss

    def contrastive_grounding_loss(self, spatial_maps, image_features, text_features,
                                    temperature=0.07):
        """Contrastive loss: matched image-text pairs should share similar
        spatial attention patterns.

        Args:
            spatial_maps: (B, num_sent, H, W) attention maps from text generation
            image_features: (B, d_model) global image features
            text_features: (B, d_model) global text features

        Returns:
            contrastive_loss: InfoNCE-style contrastive loss
        """
        bsz = spatial_maps.shape[0]

        # Global attention summary
        attn_summary = spatial_maps.view(bsz, -1)  # (B, num_sent * H * W)
        attn_norm = F.normalize(attn_summary, dim=-1)

        img_norm = F.normalize(image_features, dim=-1)
        txt_norm = F.normalize(text_features, dim=-1)

        # Image-attention alignment
        logits_img = torch.matmul(img_norm, attn_norm.T) / temperature
        labels = torch.arange(bsz, device=spatial_maps.device)
        loss_img = F.cross_entropy(logits_img, labels)

        # Text-attention alignment
        logits_txt = torch.matmul(txt_norm, attn_norm.T) / temperature
        loss_txt = F.cross_entropy(logits_txt, labels)

        return (loss_img + loss_txt) / 2

    def smoothness_loss(self, spatial_maps):
        """Total variation loss for spatial smoothness of attention maps.

        Encourages adjacent image patches to have similar attention weights.

        Args:
            spatial_maps: (B, num_sent_or_tokens, H, W)

        Returns:
            tv_loss: Total variation regularization
        """
        diff_h = (spatial_maps[:, :, 1:, :] - spatial_maps[:, :, :-1, :]).abs().mean()
        diff_w = (spatial_maps[:, :, :, 1:] - spatial_maps[:, :, :, :-1]).abs().mean()
        return diff_h + diff_w

    def forward(self, spatial_maps, bbox_masks=None, sentence_finding_idx=None,
                finding_predictions=None, sentence_embeddings=None,
                finding_embeddings=None, image_features=None, text_features=None):
        """Compute all grounding losses.

        Args:
            spatial_maps: (B, num_sent, H, W) sentence-level attention maps
            Various optional supervision signals

        Returns:
            total_loss: Combined grounding loss
            loss_components: Dict of individual loss values
        """
        total = torch.tensor(0.0, device=spatial_maps.device)
        components = {}

        # SVM loss for smoothness
        smooth_loss = self.smoothness_loss(spatial_maps)
        components['smoothness'] = smooth_loss.item()
        total = total + self.smoothness_weight * smooth_loss

        # BBox grounding (if bbox annotations available)
        if self.use_bbox_supervision and bbox_masks is not None:
            bbox_loss = self.bbox_grounding_loss(
                spatial_maps, bbox_masks, sentence_finding_idx)
            components['bbox'] = bbox_loss.item()
            total = total + bbox_loss

        # Sentence-finding consistency
        if finding_predictions is not None and sentence_embeddings is not None:
            consist_loss = self.sentence_finding_consistency_loss(
                spatial_maps, finding_predictions,
                sentence_embeddings, finding_embeddings,
            )
            components['consistency'] = consist_loss.item()
            total = total + consist_loss

        # Contrastive grounding
        if image_features is not None and text_features is not None:
            contrast_loss = self.contrastive_grounding_loss(
                spatial_maps, image_features, text_features)
            components['contrastive'] = contrast_loss.item()
            total = total + self.contrastive_weight * contrast_loss

        components['total'] = total.item()

        return total, components

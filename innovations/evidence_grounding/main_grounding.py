"""
Training entry point for GroundingLAMRG (Direction 3).

Usage:
    python -m innovations.evidence_grounding.main_grounding \
        --cfg config/iu_resnet.yml \
        --expe_name grounding_v1 \
        --label_loss --rank_loss \
        --use_bbox_supervision

Key arguments:
    --use_bbox_supervision: Enable weak bbox supervision for grounding
    --mc_samples: Number of MC Dropout samples for uncertainty
    --patch_size: Vision encoder patch size for spatial attention
    --grounding_weight: Weight for grounding loss
    --smoothness_weight: Weight for attention smoothness
"""

import sys
import os
import argparse
import logging

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.opts import prepare_dirs
from config.config import load_config
from modules.tokenizers import Tokenizer
from modules.dataloaders import create_dataloader
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.metrics import compute_metrics
from innovations.evidence_grounding.grounding_lamrg import GroundingLAMRG
from innovations.evidence_grounding.grounding_loss import GroundingLoss


class GroundingTrainer:
    """Custom trainer for GroundingLAMRG with evidence alignment losses."""

    def __init__(self, args, model, train_dl, val_dl, test_dl, optimizer, lr_scheduler):
        self.args = args
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Losses
        from modules.loss import LanguageModelCriterion, RankingLoss
        self.lm_criterion = LanguageModelCriterion()
        self.ranking_loss = RankingLoss()
        self.label_criterion = torch.nn.BCEWithLogitsLoss()
        self.grounding_loss = GroundingLoss(
            use_bbox_supervision=getattr(args, 'use_bbox_supervision', False),
            smoothness_weight=getattr(args, 'smoothness_weight', 0.01),
            contrastive_weight=getattr(args, 'contrastive_weight', 0.1),
        )
        self.grounding_weight = getattr(args, 'grounding_weight', 0.1)

        self.model = self.model.to(self.device)

    def train(self):
        logging.info(f"Training GroundingLAMRG for {self.args.epochs} epochs")
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self._train_epoch(epoch)
            logging.info(f"Epoch {epoch}/{self.args.epochs} - Loss: {train_loss:.4f}")
            self._validate(epoch)
            self.lr_scheduler.step()

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        from tqdm import tqdm

        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in \
                enumerate(tqdm(self.train_dl, desc=f'Epoch {epoch}')):
            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            results = self.model(images, reports_ids, labels, mode='train')

            # Language model loss
            lm_loss = self.lm_criterion(
                results['output'][:, :-1],
                reports_ids[:, 1:],
                reports_masks[:, 1:],
            ).mean()

            # Label loss
            label_loss = self.label_criterion(results['vis_labels'], labels)

            # Ranking loss
            if results.get('z_txt') is not None and results.get('z_img') is not None:
                rank_loss = self.ranking_loss(
                    results['z_img'], results['z_txt'], labels)
            else:
                rank_loss = torch.tensor(0.0, device=self.device)

            # Grounding loss
            if results.get('sentence_maps') is not None:
                ground_loss, _ = self.grounding_loss(
                    spatial_maps=results['sentence_maps'],
                    finding_predictions=results.get('finding_scores'),
                    sentence_embeddings=results['z_img'].unsqueeze(1),
                    finding_embeddings=self.model.finding_embed.weight,
                    image_features=results['z_img'],
                    text_features=results.get('z_txt'),
                )
            else:
                ground_loss = torch.tensor(0.0, device=self.device)

            loss = lm_loss + 0.1 * label_loss + 0.1 * rank_loss + \
                   self.grounding_weight * ground_loss

            loss.backward()
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dl)

    def _validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            val_res, val_gts = [], []
            for images_id, images, reports_ids, reports_masks, labels in self.val_dl:
                images = images.to(self.device)
                results = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(
                    results['output'].cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(
                    reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            metrics = compute_metrics(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)},
            )
            logging.info(f"Val metrics: {metrics}")


def main():
    parser = argparse.ArgumentParser(description='GroundingLAMRG: Evidence Grounding')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--expe_name', type=str, required=True)
    parser.add_argument('--version', type=int, default=300)
    parser.add_argument('--label_loss', action='store_true', default=True)
    parser.add_argument('--rank_loss', action='store_true', default=True)
    parser.add_argument('--use_bbox_supervision', action='store_true')
    parser.add_argument('--mc_samples', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--grounding_weight', type=float, default=0.1)
    parser.add_argument('--smoothness_weight', type=float, default=0.01)
    parser.add_argument('--contrastive_weight', type=float, default=0.1)

    args = parser.parse_args()

    cfg = load_config(args.cfg)
    for k, v in cfg.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    for k, v in vars(args).items():
        if v is not None:
            setattr(args, k, v)

    args = prepare_dirs(args)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    tokenizer = Tokenizer(args)
    train_dl = create_dataloader(args, tokenizer, split='train')
    val_dl = create_dataloader(args, tokenizer, split='val')
    test_dl = create_dataloader(args, tokenizer, split='test')

    model = GroundingLAMRG(args, tokenizer)
    logging.info(str(model))

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = GroundingTrainer(
        args=args, model=model,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
        optimizer=optimizer, lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == '__main__':
    main()

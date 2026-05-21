"""
Training entry point for TemporalLAMRG (Direction 5).

Usage:
    python -m innovations.temporal_progression.main_temporal \
        --cfg config/mimic_resnet.yml \
        --expe_name temporal_v1 \
        --label_loss --rank_loss \
        --max_time_delta 365

Key arguments:
    --max_time_delta: Maximum days between paired studies
    --min_time_delta: Minimum days between paired studies
    --progression_weight: Weight for progression classification loss
    --consistency_weight: Weight for temporal consistency loss
    --smoothness_weight: Weight for severity smoothness loss
    --report_alignment_weight: Weight for report-label alignment loss
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
from modules.trainer import Trainer as BaseTrainer
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.metrics import compute_metrics
from innovations.temporal_progression.temporal_lamrg import TemporalLAMRG
from innovations.temporal_progression.temporal_loss import TemporalConsistencyLoss


def temporal_compute_loss(outputs, reports_ids, reports_masks, labels, args, **kwargs):
    """Extended loss with temporal progression components.

    Expects outputs tuple: (output, vis_labels, txt_labels, z_img, z_txt,
                            progression_logits, severity_change, temporal_features)
    """
    from modules.loss import LanguageModelCriterion, RankingLoss

    output = outputs[0]
    vis_labels = outputs[1]
    txt_labels = outputs[2]
    z_img = outputs[3]
    z_txt = outputs[4]
    progression_logits = outputs[5]
    severity_change = outputs[6]
    temporal_features = outputs[7]

    # Standard losses
    lm_criterion = LanguageModelCriterion()
    lm_loss = lm_criterion(output[:, :-1], reports_ids[:, 1:],
                            reports_masks[:, 1:]).mean()

    label_loss = torch.tensor(0.0, device=output.device)
    match_loss = torch.tensor(0.0, device=output.device)

    if getattr(args, 'label_loss', False):
        label_criterion = torch.nn.BCEWithLogitsLoss()
        label_loss = label_criterion(vis_labels, labels)

    if getattr(args, 'rank_loss', False) and z_img is not None and z_txt is not None:
        ranking_loss = RankingLoss()
        match_loss = ranking_loss(z_img, z_txt, labels)

    # Temporal losses
    temporal_loss_fn = TemporalConsistencyLoss(
        num_labels=args.num_labels,
        consistency_weight=getattr(args, 'consistency_weight', 0.5),
        progression_weight=getattr(args, 'progression_weight', 1.0),
        smoothness_weight=getattr(args, 'smoothness_weight', 0.3),
        report_alignment_weight=getattr(args, 'report_alignment_weight', 0.3),
    )

    label_progression = kwargs.get('label_progression', torch.zeros_like(labels))
    time_deltas = kwargs.get('time_deltas', None)

    temporal_loss, temporal_components = temporal_loss_fn(
        progression_logits=progression_logits,
        label_progression=label_progression,
        temporal_features=temporal_features,
        time_deltas=time_deltas,
        severity_changes=severity_change,
    )

    total = lm_loss + 0.1 * label_loss + 0.1 * match_loss + temporal_loss.to(output.device)

    return total


def main():
    parser = argparse.ArgumentParser(description='TemporalLAMRG: Longitudinal Modeling')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--expe_name', type=str, required=True)
    parser.add_argument('--version', type=int, default=500)
    parser.add_argument('--label_loss', action='store_true', default=True)
    parser.add_argument('--rank_loss', action='store_true', default=True)

    # Temporal-specific
    parser.add_argument('--max_time_delta', type=int, default=365)
    parser.add_argument('--min_time_delta', type=int, default=1)
    parser.add_argument('--progression_weight', type=float, default=1.0)
    parser.add_argument('--consistency_weight', type=float, default=0.5)
    parser.add_argument('--smoothness_weight', type=float, default=0.3)
    parser.add_argument('--report_alignment_weight', type=float, default=0.3)

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

    model = TemporalLAMRG(args, tokenizer)
    logging.info(str(model))

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = BaseTrainer(
        model=model,
        criterion=temporal_compute_loss,
        metric_ftns=compute_metrics,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
    )

    trainer.train()


if __name__ == '__main__':
    main()

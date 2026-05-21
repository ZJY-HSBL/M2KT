"""
Training entry point for AgentLAMRG (Direction 2).

Usage:
    python -m innovations.agent_refinement.main_agent \
        --cfg config/iu_resnet.yml \
        --expe_name agent_v1 \
        --max_refine_steps 3 \
        --use_dpo \
        --dpo_beta 0.1

Key arguments:
    --max_refine_steps: Max refinement iterations (default: 3)
    --use_dpo: Enable Direct Preference Optimization
    --dpo_beta: DPO temperature parameter
    --dpo_weight: Weight of DPO loss in total loss
    --consistency_weight: Weight of consistency auxiliary loss
    --completeness_weight: Weight of completeness auxiliary loss
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
from modules.loss import LanguageModelCriterion
from modules.metrics import compute_metrics
from innovations.agent_refinement.agent_lamrg import AgentLAMRG
from innovations.agent_refinement.rl_trainer import DPOTrainer


def main():
    parser = argparse.ArgumentParser(description='AgentLAMRG: Iterative Self-Refinement')

    parser.add_argument('--cfg', type=str, required=True, help='Config file path')
    parser.add_argument('--expe_name', type=str, required=True)
    parser.add_argument('--version', type=int, default=200)

    # Agent-specific
    parser.add_argument('--max_refine_steps', type=int, default=3)
    parser.add_argument('--refiner_layers', type=int, default=3)
    parser.add_argument('--use_dpo', action='store_true', default=True)
    parser.add_argument('--dpo_beta', type=float, default=0.1)
    parser.add_argument('--dpo_weight', type=float, default=0.5)
    parser.add_argument('--consistency_weight', type=float, default=0.1)
    parser.add_argument('--completeness_weight', type=float, default=0.1)

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.cfg)
    for k, v in cfg.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    for k, v in vars(args).items():
        if v is not None:
            setattr(args, k, v)

    args = prepare_dirs(args)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )

    # Build components
    tokenizer = Tokenizer(args)
    train_dl = create_dataloader(args, tokenizer, split='train')
    val_dl = create_dataloader(args, tokenizer, split='val')
    test_dl = create_dataloader(args, tokenizer, split='test')

    model = AgentLAMRG(args, tokenizer)
    logging.info(str(model))

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    criterion = LanguageModelCriterion()

    trainer = DPOTrainer(
        model=model,
        criterion=criterion,
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

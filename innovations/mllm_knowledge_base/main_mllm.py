"""
Training entry point for MLLM-LAMRG (Direction 1).

Usage:
    python -m innovations.mllm_knowledge_base.main_mllm \
        --cfg config/iu_resnet.yml \
        --expe_name mllm_v1 \
        --label_loss --rank_loss \
        --vlm_type biomedclip \
        --freeze_vlm

Key additions over original main.py:
    --vlm_type: Medical VLM backbone (biomedclip, biovil, chexzero, cnn)
    --freeze_vlm: Whether to freeze VLM weights
    --num_cross_slots: Number of cross-disease memory slots
    --num_memory_heads: Attention heads for memory operations
"""

import sys
import os
import argparse
import logging

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.opts import parse_args, prepare_dirs
from config.config import load_config
from modules.tokenizers import Tokenizer
from modules.dataloaders import create_dataloader
from modules.trainer import Trainer
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.loss import compute_loss
from modules.metrics import compute_metrics
from innovations.mllm_knowledge_base.mllm_lamrg import MLLMLAMRG


def main():
    parser = argparse.ArgumentParser(description='MLLM-LAMRG: Medical VLM Knowledge Base')
    parser.add_argument('--cfg', type=str, required=True, help='Config file path')
    parser.add_argument('--expe_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--label_loss', action='store_true', help='Enable label prediction loss')
    parser.add_argument('--rank_loss', action='store_true', help='Enable ranking loss')
    parser.add_argument('--vlm_type', type=str, default='biomedclip',
                        choices=['biomedclip', 'biovil', 'chexzero', 'cnn'],
                        help='Medical VLM backbone type')
    parser.add_argument('--freeze_vlm', action='store_true', default=True,
                        help='Freeze VLM backbone weights')
    parser.add_argument('--num_cross_slots', type=int, default=32,
                        help='Number of cross-disease memory slots')
    parser.add_argument('--num_memory_heads', type=int, default=8,
                        help='Attention heads for memory operations')
    parser.add_argument('--num_slots', type=int, default=256,
                        help='Number of instance-level memory slots')

    # Inherit standard arguments
    parser.add_argument('--version', type=int, default=100, help='MLLM model version')
    args = parser.parse_args()

    # Load base config and override
    cfg = load_config(args.cfg)
    for k, v in vars(args).items():
        if v is not None and k != 'cfg':
            setattr(args, k, v)

    # Merge config values
    for k, v in cfg.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)

    args = prepare_dirs(args)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )

    # Build tokenizer
    tokenizer = Tokenizer(args)

    # Build dataloaders
    train_dataloader = create_dataloader(args, tokenizer, split='train')
    val_dataloader = create_dataloader(args, tokenizer, split='val')
    test_dataloader = create_dataloader(args, tokenizer, split='test')

    # Build model
    model = MLLMLAMRG(args, tokenizer)
    logging.info(str(model))

    # Build optimizer and scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # Build trainer
    trainer = Trainer(
        model=model,
        criterion=compute_loss,
        metric_ftns=compute_metrics,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
    )

    trainer.train()


if __name__ == '__main__':
    main()

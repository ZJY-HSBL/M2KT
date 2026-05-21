"""
Training entry point for GNNLAMRG (Direction 4).

Usage:
    python -m innovations.gnn_ontology.main_gnn \
        --cfg config/iu_resnet.yml \
        --expe_name gnn_v1 \
        --label_loss --rank_loss \
        --ontology_source builtin

Key arguments:
    --ontology_source: 'builtin' or path to external ontology JSON
    --gnn_heads: Number of GAT attention heads
    --graph_reg_weight: Weight for graph adjacency regularization
    --fusion_decision_weight: Weight for fusion decision loss
    --num_memory_heads: Attention heads for memory operations
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
from modules.trainer import Trainer
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.metrics import compute_metrics
from innovations.gnn_ontology.gnn_lamrg import GNNLAMRG


def gnn_compute_loss(output, reports_ids, reports_masks, labels,
                     vis_label=None, txt_label=None,
                     z_img=None, z_txt=None, args={}, aux_losses=None,
                     similarity_function='dot'):
    """Extended loss function that includes GNN auxiliary losses."""
    from modules.loss import LanguageModelCriterion, RankingLoss

    lm_criterion = LanguageModelCriterion()
    lm_loss = lm_criterion(output[:, :-1], reports_ids[:, 1:],
                            reports_masks[:, 1:]).mean()

    label_loss = torch.tensor(0.0, device=output.device)
    match_loss = torch.tensor(0.0, device=output.device)

    if args.label_loss:
        label_criterion = torch.nn.BCEWithLogitsLoss()
        label_loss = label_criterion(vis_label, labels)

    if args.rank_loss and z_img is not None and z_txt is not None:
        ranking_loss = RankingLoss()
        match_loss = ranking_loss(z_img, z_txt, labels, similarity_function)

    # GNN auxiliary losses
    graph_reg_loss = torch.tensor(0.0, device=output.device)
    if aux_losses is not None:
        graph_reg_loss = aux_losses.get('graph_reg_loss', torch.tensor(0.0, device=output.device))

    graph_reg_weight = getattr(args, 'graph_reg_weight', 0.01)

    return lm_loss + 0.1 * label_loss + 0.1 * match_loss + graph_reg_weight * graph_reg_loss


def main():
    parser = argparse.ArgumentParser(description='GNNLAMRG: GNN + Ontology Enhancement')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--expe_name', type=str, required=True)
    parser.add_argument('--version', type=int, default=400)
    parser.add_argument('--label_loss', action='store_true', default=True)
    parser.add_argument('--rank_loss', action='store_true', default=True)
    parser.add_argument('--ontology_source', type=str, default='builtin')
    parser.add_argument('--gnn_heads', type=int, default=4)
    parser.add_argument('--graph_reg_weight', type=float, default=0.01)
    parser.add_argument('--fusion_decision_weight', type=float, default=0.01)
    parser.add_argument('--num_memory_heads', type=int, default=8)

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

    model = GNNLAMRG(args, tokenizer)
    logging.info(str(model))

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = Trainer(
        model=model,
        criterion=gnn_compute_loss,
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

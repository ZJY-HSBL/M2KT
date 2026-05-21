"""
RL Trainer with Direct Preference Optimization (DPO) for Agent-based report generation.

DPO eliminates the need for a separate reward model by directly optimizing
the policy from pairwise preference data. Preferences are constructed as:
    - Preferred: refined report (after consistency check + completeness verification)
    - Dispreferred: initial draft report

Additional training modes:
    - RLCF: Reinforcement Learning from Clinical Feedback
    - Self-play: Model generates multiple candidates, best is used as positive example
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm

from modules.trainer import BaseTrainer, METRICS


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss for report refinement.

    DPO directly optimizes the policy to prefer refined reports over drafts:
        L_DPO = -log σ(β * (log π(y_w|x) / π_ref(y_w|x) - log π(y_l|x) / π_ref(y_l|x)))

    where y_w is the refined (preferred) report and y_l is the draft (dispreferred).
    """

    def __init__(self, beta=0.1, label_smoothing=0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, policy_refined_logps, policy_draft_logps,
                ref_refined_logps, ref_draft_logps):
        """Compute DPO loss for a batch.

        Args:
            policy_refined_logps: (B,) log-prob of refined report under current policy
            policy_draft_logps: (B,) log-prob of draft report under current policy
            ref_refined_logps: (B,) log-prob of refined report under reference model
            ref_draft_logps: (B,) log-prob of draft report under reference model

        Returns:
            loss: Scalar DPO loss
            accuracy: Fraction where refined > draft preference is correct
        """
        policy_ratio = policy_refined_logps - policy_draft_logps
        ref_ratio = ref_refined_logps - ref_draft_logps

        logits = self.beta * (policy_ratio - ref_ratio)
        loss = -F.logsigmoid(logits).mean()

        accuracy = (logits > 0).float().mean()
        return loss, accuracy


class DPOTrainer(BaseTrainer):
    """Trainer with DPO for agent-based iterative refinement.

    Extends BaseTrainer with:
        - DPO loss on draft vs refined report pairs
        - Consistency and completeness auxiliary losses
        - Reference model management for DPO
        - Multi-stage training (pretrain generator -> train refiner -> DPO finetune)
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, args,
                 lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super().__init__(model, criterion, metric_ftns, optimizer, args)

        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # DPO-specific
        self.dpo_loss = DPOLoss(
            beta=getattr(args, 'dpo_beta', 0.1),
        )
        self.dpo_weight = getattr(args, 'dpo_weight', 0.5)
        self.consistency_weight = getattr(args, 'consistency_weight', 0.1)
        self.completeness_weight = getattr(args, 'completeness_weight', 0.1)

        # Reference model (frozen copy for DPO)
        self.ref_model = None
        if getattr(args, 'use_dpo', True):
            self._init_ref_model()

    def _init_ref_model(self):
        """Create frozen reference model for DPO."""
        import copy
        self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

    def _compute_log_prob(self, logits, targets, mask):
        """Compute sequence log-probability under the model.

        Args:
            logits: (B, seq_len, vocab) output log-probabilities
            targets: (B, seq_len) target token ids
            mask: (B, seq_len) target mask

        Returns:
            seq_log_probs: (B,) average log-prob per token
        """
        targets = targets[:, :logits.size(1)]
        mask = mask[:, :logits.size(1)]

        # Gather log-prob at target positions
        token_log_probs = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * mask
        seq_log_probs = token_log_probs.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return seq_log_probs

    def _train_epoch(self, epoch):
        train_loss = 0
        train_dpo_loss = 0
        train_consistency_loss = 0
        train_completeness_loss = 0

        self.model.train()
        t = tqdm(self.train_dataloader, ncols=100)

        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(t):
            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward through agent pipeline
            results = self.model(images, reports_ids, labels, mode='train')

            # ---- LM Loss on draft ----
            lm_loss = self.criterion(
                results['draft_output'][:, :-1],
                reports_ids[:, 1:],
                reports_masks[:, 1:],
            ).mean()

            # ---- LM Loss on refined output ----
            refined_lm_loss = self.criterion(
                results['refined_output'][:, :-1],
                reports_ids[:, 1:],
                reports_masks[:, 1:],
            ).mean()

            # ---- DPO Loss ----
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_results = self.ref_model(images, reports_ids, labels, mode='train')

                policy_refined_lp = self._compute_log_prob(
                    results['refined_output'], reports_ids, reports_masks)
                policy_draft_lp = self._compute_log_prob(
                    results['draft_output'], reports_ids, reports_masks)
                ref_refined_lp = self._compute_log_prob(
                    ref_results['refined_output'], reports_ids, reports_masks)
                ref_draft_lp = self._compute_log_prob(
                    ref_results['draft_output'], reports_ids, reports_masks)

                dpo_loss, dpo_acc = self.dpo_loss(
                    policy_refined_lp, policy_draft_lp,
                    ref_refined_lp, ref_draft_lp,
                )
            else:
                dpo_loss = torch.tensor(0.0, device=self.device)
                dpo_acc = torch.tensor(0.0, device=self.device)

            # ---- Auxiliary losses ----
            # Encourage consistency (low contradiction)
            consistency_loss = results['contra_scores'].mean()

            # Encourage completeness (high coverage of positive labels)
            completeness_loss = 1.0 - results['completeness_score'].mean()

            # ---- Total loss ----
            total_loss = (
                lm_loss +
                refined_lm_loss +
                self.dpo_weight * dpo_loss +
                self.consistency_weight * consistency_loss +
                self.completeness_weight * completeness_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            train_loss += total_loss.item()
            train_dpo_loss += dpo_loss.item()
            train_consistency_loss += consistency_loss.item()
            train_completeness_loss += completeness_loss.item()

            t.set_description(
                f'L:{total_loss.item():.3f} '
                f'DPO:{dpo_loss.item():.3f} '
                f'Cons:{consistency_loss.item():.3f} '
                f'Comp:{completeness_loss.item():.3f}'
            )

        n = len(self.train_dataloader)
        log = {
            'train_loss': train_loss / n,
            'train_dpo_loss': train_dpo_loss / n,
            'train_consistency_loss': train_consistency_loss / n,
            'train_completeness_loss': train_completeness_loss / n,
        }

        # Validation
        ilog = self._test_step(epoch, 0, 'val')
        log.update(**ilog)
        ilog = self._test_step(epoch, 0, 'test')
        log.update(**ilog)

        self.lr_scheduler.step()
        return log

    def _test_step(self, epoch, iters=0, mode='test'):
        ilog = {}
        self.model.eval()
        data_loader = self.val_dataloader if mode == 'val' else self.test_dataloader

        with torch.no_grad():
            val_gts, val_res = [], []
            t = tqdm(data_loader, ncols=80)
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(t):
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)
                labels = labels.to(self.device)

                results = self.model(images, mode='sample')

                # Use refined output for evaluation
                reports = self.model.tokenizer.decode_batch(
                    results['refined_output'].cpu().numpy()
                )
                ground_truths = self.model.tokenizer.decode_batch(
                    reports_ids[:, 1:].cpu().numpy()
                )
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)},
            )
            ilog.update(**{f'{mode}_' + k: v for k, v in val_met.items()})

        return ilog

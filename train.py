"""
IMFER Training Script — Section IV-C.

Usage:
    python train.py --config config/default.yaml --dataset iemocap --data_dir data/iemocap
    python train.py --config config/default.yaml --dataset meld --data_dir data/meld

Implementation details (paper):
    - AdamW optimizer
    - lr = 2×10⁻⁵ for pretrained encoders, 10⁻³ for new layers
    - 10% linear warmup
    - Batch size: 32 utterances
    - Early stopping: patience = 10 on validation WF1
    - 5 independent runs with different seeds
    - ~72 compute-hours per dataset on M4 Pro (wall-clock)

Loss (Eq. 7):
    L = L_CE + λ₁ L_MCS + λ₂ L_align
    λ₁ = 0.1,  λ₂ = 0.05,  τ = 0.07

MELD uses weighted cross-entropy for class imbalance.
"""

import os
import sys
import json
import argparse
import logging
import time
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from imfer.models.imfer import IMFERFromFeatures
from imfer.data.dataset import ERCDataset, collate_fn, LABEL_MAPS
from imfer.utils.metrics import compute_metrics
from imfer.utils.helpers import (
    set_seed, get_device, AverageMeter, EarlyStopping,
    setup_logging, count_parameters,
)


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """
    Linear warmup followed by linear decay (Section IV-C: 10% warmup).
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


def get_class_weights(dataset, num_classes, device):
    """
    Compute inverse-frequency class weights for weighted cross-entropy.
    Used for MELD and EmoryNLP (Section IV-A).
    """
    counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        item = dataset[i]
        labels = item['labels']
        for c in range(num_classes):
            counts[c] += (labels == c).sum().item()

    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Single training epoch."""
    model.train()
    loss_meter = AverageMeter()

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        output = model(batch)
        loss = output['loss']

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        n_valid = batch['utterance_mask'].sum().item()
        loss_meter.update(loss.item(), int(n_valid))

        if (step + 1) % 50 == 0:
            logging.info(
                f"  Epoch {epoch} | Step {step+1} | "
                f"Loss: {loss_meter.avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model on a split.
    Returns: dict with wf1, acc, mf1, per_class_f1, avg_loss, avg_mcs
    """
    model.eval()
    all_preds, all_labels, all_mcs_scores = [], [], []
    loss_meter = AverageMeter()

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        output = model(batch)
        logits = output['logits']       # (B, N, C)
        mcs = output['mcs']             # (B, N, 3)
        labels = batch['labels']        # (B, N)
        utt_mask = batch['utterance_mask']

        if 'loss' in output:
            n_valid = utt_mask.sum().item()
            loss_meter.update(output['loss'].item(), int(n_valid))

        valid = utt_mask.reshape(-1)
        preds = logits.reshape(-1, logits.size(-1))[valid].argmax(dim=-1)
        labs = labels.reshape(-1)[valid]
        m = mcs.reshape(-1, 3)[valid]

        all_preds.append(preds.cpu())
        all_labels.append(labs.cpu())
        all_mcs_scores.append(m.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_mcs_scores = torch.cat(all_mcs_scores).numpy()

    metrics = compute_metrics(all_labels, all_preds)
    metrics['avg_loss'] = loss_meter.avg
    metrics['avg_mcs'] = all_mcs_scores.mean(axis=0).tolist()

    return metrics


def train_single_run(cfg, run_idx, device):
    """
    Execute a single training run.

    Returns: dict with best val metrics and test metrics
    """
    seed = cfg['training']['seed'] + run_idx
    set_seed(seed)
    logging.info(f"\n{'='*60}")
    logging.info(f"Run {run_idx+1}/{cfg['training']['num_runs']} | Seed: {seed}")
    logging.info(f"{'='*60}")

    dataset_name = cfg['dataset']
    data_dir = cfg['data_dir']
    ds_cfg = cfg['datasets'][dataset_name]
    num_classes = ds_cfg['num_classes']

    # ── Data ──────────────────────────────────────────────────────────
    train_set = ERCDataset(data_dir, "train", dataset_name)
    val_set = ERCDataset(data_dir, "val", dataset_name)
    test_set = ERCDataset(data_dir, "test", dataset_name)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    # Note: batch_size=1 means 1 conversation per batch.
    # Each conversation has N utterances ≈ batch of ~32 utterances on average.

    # ── Model ─────────────────────────────────────────────────────────
    model = IMFERFromFeatures(
        num_classes=num_classes,
        text_dim=cfg['model']['text_dim'],
        audio_dim=cfg['model']['audio_dim'],
        visual_dim=cfg['model']['visual_dim'],
        d_k=cfg['model']['d_k'],
        d_model=cfg['model']['d_model'],
        context_window=cfg['model']['context_window'],
        gat_heads=cfg['model']['gat_heads'],
        transformer_layers=cfg['model']['transformer_layers'],
        transformer_heads=cfg['model']['transformer_heads'],
        dropout=cfg['model']['dropout'],
        lambda_mcs=cfg['training']['lambda_mcs'],
        lambda_align=cfg['training']['lambda_align'],
        tau=cfg['training']['tau'],
        mcs_epsilon=cfg['model']['mcs_epsilon'],
    ).to(device)

    if run_idx == 0:
        params = count_parameters(model)
        logging.info(f"Model parameters: {params['total_M']:.1f}M total, "
                     f"{params['trainable_M']:.1f}M trainable")

    # ── Optimizer: dual learning rates (Section IV-C) ─────────────────
    # Paper: lr = 2×10⁻⁵ for pretrained encoders, 10⁻³ for new layers
    # Since IMFERFromFeatures has no encoder parameters, all are "new".
    # When using IMFER (with encoders), split param groups accordingly.
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['lr_new'],
        weight_decay=cfg['training']['weight_decay'],
    )

    # ── Scheduler: 10% linear warmup ──────────────────────────────────
    total_steps = cfg['training']['epochs'] * len(train_loader)
    warmup_steps = int(total_steps * cfg['training']['warmup_ratio'])
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # ── Early stopping: patience = 10 on validation WF1 ──────────────
    ckpt_path = os.path.join(
        cfg['logging']['checkpoint_dir'],
        f"{dataset_name}_run{run_idx}_best.pt"
    )
    early_stop = EarlyStopping(
        patience=cfg['training']['patience'],
        checkpoint_path=ckpt_path,
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_metrics = None

    for epoch in range(1, cfg['training']['epochs'] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     device, epoch)
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        logging.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val WF1: {val_metrics['wf1']:.2f}% | "
            f"Val Acc: {val_metrics['acc']:.2f}% | "
            f"Val MF1: {val_metrics['mf1']:.2f}% | "
            f"MCS: [{val_metrics['avg_mcs'][0]:.3f}, "
            f"{val_metrics['avg_mcs'][1]:.3f}, "
            f"{val_metrics['avg_mcs'][2]:.3f}] | "
            f"Time: {elapsed:.1f}s"
        )

        if early_stop.step(val_metrics['wf1'], model):
            logging.info(f"Early stopping at epoch {epoch}")
            break

        if val_metrics['wf1'] == early_stop.best_score:
            best_val_metrics = val_metrics

    # ── Test evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    logging.info(f"\nTest Results (Run {run_idx+1}):")
    logging.info(f"  WF1:  {test_metrics['wf1']:.2f}%")
    logging.info(f"  Acc:  {test_metrics['acc']:.2f}%")
    logging.info(f"  MF1:  {test_metrics['mf1']:.2f}%")
    logging.info(f"  MCS:  {test_metrics['avg_mcs']}")

    return {
        'best_val': best_val_metrics,
        'test': test_metrics,
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description="IMFER Training")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["iemocap", "meld", "emorynlp"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_runs", type=int, default=None,
                        help="Override number of runs (default: 5)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['dataset'] = args.dataset
    cfg['data_dir'] = args.data_dir
    if args.num_runs is not None:
        cfg['training']['num_runs'] = args.num_runs

    # Setup
    setup_logging(cfg['logging']['log_dir'], f"{args.dataset}_train")
    device = get_device()
    logging.info(f"Config: {json.dumps(cfg, indent=2)}")

    # ── Run N independent experiments ─────────────────────────────────
    num_runs = cfg['training']['num_runs']
    all_results = []

    for run_idx in range(num_runs):
        result = train_single_run(cfg, run_idx, device)
        all_results.append(result)

    # ── Aggregate statistics across runs ──────────────────────────────
    test_wf1s = [r['test']['wf1'] for r in all_results]
    test_accs = [r['test']['acc'] for r in all_results]
    test_mf1s = [r['test']['mf1'] for r in all_results]

    mean_wf1 = np.mean(test_wf1s)
    std_wf1 = np.std(test_wf1s)
    ci95_wf1 = 1.96 * std_wf1 / np.sqrt(num_runs)

    logging.info(f"\n{'='*60}")
    logging.info(f"FINAL RESULTS ({args.dataset.upper()}, {num_runs} runs)")
    logging.info(f"{'='*60}")
    logging.info(f"WF1:  {mean_wf1:.2f} ± {std_wf1:.2f}%  (95% CI: ±{ci95_wf1:.2f}%)")
    logging.info(f"Acc:  {np.mean(test_accs):.2f} ± {np.std(test_accs):.2f}%")
    logging.info(f"MF1:  {np.mean(test_mf1s):.2f} ± {np.std(test_mf1s):.2f}%")

    # Per-class F1
    if all_results[0]['test']['per_class_f1']:
        per_class = np.array([r['test']['per_class_f1'] for r in all_results])
        label_names = cfg['datasets'][args.dataset]['labels']
        logging.info("\nPer-class F1 (mean ± std):")
        for c, name in enumerate(label_names):
            if c < per_class.shape[1]:
                logging.info(f"  {name:12s}: {per_class[:, c].mean():.1f} ± "
                             f"{per_class[:, c].std():.1f}%")

    # Save results
    results_path = os.path.join(cfg['logging']['log_dir'],
                                f"{args.dataset}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'num_runs': num_runs,
            'mean_wf1': mean_wf1,
            'std_wf1': std_wf1,
            'ci95_wf1': ci95_wf1,
            'mean_acc': float(np.mean(test_accs)),
            'mean_mf1': float(np.mean(test_mf1s)),
            'per_run': all_results,
        }, f, indent=2, default=str)

    logging.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

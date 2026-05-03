"""
IMFER Evaluation & Analysis Script — Sections V-A through V-I.

Usage:
    # Basic evaluation
    python evaluate.py --checkpoint checkpoints/best.pt --dataset iemocap --data_dir data/iemocap

    # AOPC faithfulness analysis (Section V-F)
    python evaluate.py --checkpoint checkpoints/best.pt --dataset iemocap --data_dir data/iemocap --aopc

    # Noise sensitivity analysis (Section V-G)
    python evaluate.py --checkpoint checkpoints/best.pt --dataset iemocap --data_dir data/iemocap \
        --noise_sigma 0.1 0.3 0.6 1.0

    # Missing modality robustness (Section V-G, Table VII)
    python evaluate.py --checkpoint checkpoints/best.pt --dataset iemocap --data_dir data/iemocap \
        --missing_modality

    # Cross-dataset transfer (Section V-H)
    python evaluate.py --checkpoint checkpoints/iemocap_best.pt --dataset meld --data_dir data/meld \
        --zero_shot

    # Full analysis (all of the above)
    python evaluate.py --checkpoint checkpoints/best.pt --dataset iemocap --data_dir data/iemocap --full
"""

import os
import json
import argparse
import logging

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from imfer.models.imfer import IMFERFromFeatures
from imfer.data.dataset import ERCDataset, collate_fn
from imfer.utils.metrics import (
    compute_metrics, compute_aopc, compute_faithfulness,
    compute_confusion_matrix, compute_classification_report,
)
from imfer.utils.helpers import set_seed, get_device, setup_logging


# ═══════════════════════════════════════════════════════════════════════════
# Core evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_evaluation(model, dataloader, device, label_names=None):
    """
    Standard evaluation — WF1, Acc, MF1, per-class F1, MCS analysis.
    Corresponds to Table III, Table IV.
    """
    model.eval()
    all_preds, all_labels, all_mcs = [], [], []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch)
        valid = batch['utterance_mask'].reshape(-1)
        preds = out['logits'].reshape(-1, out['logits'].size(-1))[valid].argmax(-1)
        all_preds.append(preds.cpu())
        all_labels.append(batch['labels'].reshape(-1)[valid].cpu())
        all_mcs.append(out['mcs'].reshape(-1, 3)[valid].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_mcs = torch.cat(all_mcs).numpy()

    metrics = compute_metrics(all_labels, all_preds)

    logging.info("\n" + "=" * 50)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 50)
    logging.info(f"WF1:  {metrics['wf1']:.2f}%")
    logging.info(f"Acc:  {metrics['acc']:.2f}%")
    logging.info(f"MF1:  {metrics['mf1']:.2f}%")

    # Per-class F1 (Table IV)
    if label_names:
        logging.info("\nPer-class F1:")
        for i, name in enumerate(label_names):
            if i < len(metrics['per_class_f1']):
                logging.info(f"  {name:12s}: {metrics['per_class_f1'][i]:.1f}%")

    # MCS distribution (Section V-E, Fig. 8)
    avg_mcs = all_mcs.mean(axis=0)
    logging.info(f"\nAverage MCS: text={avg_mcs[0]:.3f}, "
                 f"audio={avg_mcs[1]:.3f}, visual={avg_mcs[2]:.3f}")

    # Per-class MCS (Fig. 8)
    if label_names:
        logging.info("\nPer-class MCS:")
        for c, name in enumerate(label_names):
            mask = all_labels == c
            if mask.sum() > 0:
                class_mcs = all_mcs[mask].mean(axis=0)
                logging.info(f"  {name:12s}: T={class_mcs[0]:.3f}, "
                             f"A={class_mcs[1]:.3f}, V={class_mcs[2]:.3f}")

    # Confusion matrix (Fig. 11)
    _, cm_norm = compute_confusion_matrix(all_labels, all_preds)

    # Full classification report
    report = compute_classification_report(all_labels, all_preds, label_names)
    logging.info(f"\nClassification Report:\n{report}")

    return {
        'metrics': metrics,
        'avg_mcs': avg_mcs.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'mcs_scores': all_mcs.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Noise sensitivity analysis (Section V-G)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def noise_sensitivity_analysis(model, dataloader, device, sigmas):
    """
    Inject Gaussian noise N(0, σ²) into each modality's features.

    Tests MCS gating robustness to corrupted (not absent) data.
    Fig. 10: IMFER maintains competitive WF1 up to σ=0.6.
    """
    model.eval()
    modality_keys = ['text_features', 'audio_features', 'visual_features']
    modality_names = ['text', 'audio', 'visual']

    results = {}

    for sigma in sigmas:
        results[sigma] = {}

        for mod_idx, (mod_key, mod_name) in enumerate(zip(modality_keys, modality_names)):
            all_preds, all_labels = [], []

            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Inject noise into this modality only
                if mod_key in batch:
                    noise = torch.randn_like(batch[mod_key]) * sigma
                    batch[mod_key] = batch[mod_key] + noise

                out = model(batch)
                valid = batch['utterance_mask'].reshape(-1)
                preds = out['logits'].reshape(-1, out['logits'].size(-1))[valid].argmax(-1)
                all_preds.append(preds.cpu())
                all_labels.append(batch['labels'].reshape(-1)[valid].cpu())

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            wf1 = compute_metrics(all_labels, all_preds)['wf1']
            results[sigma][mod_name] = wf1

        logging.info(f"σ={sigma:.1f} | T: {results[sigma]['text']:.2f}%, "
                     f"A: {results[sigma]['audio']:.2f}%, "
                     f"V: {results[sigma]['visual']:.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Missing modality robustness (Section V-G, Table VII)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def missing_modality_analysis(model, dataloader, device):
    """
    Evaluate under missing-modality conditions (Table VII).

    Conditions: T+A+V, T+A, T+V, T only
    """
    model.eval()
    conditions = {
        'T+A+V': [],
        'T+A':   ['visual_features'],
        'T+V':   ['audio_features'],
        'T':     ['audio_features', 'visual_features'],
    }
    results = {}

    for cond_name, zero_keys in conditions.items():
        all_preds, all_labels = [], []

        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            for key in zero_keys:
                if key in batch:
                    batch[key] = torch.zeros_like(batch[key])

            out = model(batch)
            valid = batch['utterance_mask'].reshape(-1)
            preds = out['logits'].reshape(-1, out['logits'].size(-1))[valid].argmax(-1)
            all_preds.append(preds.cpu())
            all_labels.append(batch['labels'].reshape(-1)[valid].cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        wf1 = compute_metrics(all_labels, all_preds)['wf1']
        results[cond_name] = wf1
        logging.info(f"  {cond_name:6s}: WF1 = {wf1:.2f}%")

    # Deltas
    baseline = results['T+A+V']
    for cond in ['T+A', 'T+V', 'T']:
        delta = results[cond] - baseline
        logging.info(f"  Δ({cond}) = {delta:+.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Cross-dataset MCS distribution shift (Section V-H)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def mcs_distribution_analysis(model, dataloader, device, dataset_name):
    """
    Analyse MCS distribution on a given dataset.

    Section V-H: Text MCS rises in MELD (+0.12), audio MCS declines.
    This automatic re-weighting shows domain adaptation without explicit tuning.
    """
    model.eval()
    all_mcs = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch)
        valid = batch['utterance_mask'].reshape(-1)
        all_mcs.append(out['mcs'].reshape(-1, 3)[valid].cpu())

    all_mcs = torch.cat(all_mcs).numpy()
    avg = all_mcs.mean(axis=0)
    std = all_mcs.std(axis=0)

    logging.info(f"\nMCS Distribution ({dataset_name}):")
    logging.info(f"  Text:   {avg[0]:.3f} ± {std[0]:.3f}")
    logging.info(f"  Audio:  {avg[1]:.3f} ± {std[1]:.3f}")
    logging.info(f"  Visual: {avg[2]:.3f} ± {std[2]:.3f}")

    return {'mean': avg.tolist(), 'std': std.tolist()}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IMFER Evaluation")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["iemocap", "meld", "emorynlp"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")

    # Analysis flags
    parser.add_argument("--aopc", action="store_true",
                        help="Run AOPC faithfulness analysis (Section V-F)")
    parser.add_argument("--faithfulness", action="store_true",
                        help="Run faithfulness correlation (Table VI)")
    parser.add_argument("--noise_sigma", type=float, nargs="+", default=None,
                        help="Noise sensitivity sigmas (Section V-G)")
    parser.add_argument("--missing_modality", action="store_true",
                        help="Missing modality analysis (Table VII)")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Zero-shot cross-dataset evaluation (Section V-H)")
    parser.add_argument("--full", action="store_true",
                        help="Run all analyses")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg['logging']['log_dir'], f"{args.dataset}_eval")
    device = get_device()
    set_seed(cfg['training']['seed'])

    ds_cfg = cfg['datasets'][args.dataset]
    num_classes = ds_cfg['num_classes']
    label_names = ds_cfg['labels']

    # ── Load model ────────────────────────────────────────────────────
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
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    logging.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Data ──────────────────────────────────────────────────────────
    test_set = ERCDataset(args.data_dir, args.split, args.dataset)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    all_results = {}

    # ── Standard evaluation ───────────────────────────────────────────
    eval_results = run_evaluation(model, test_loader, device, label_names)
    all_results['standard'] = eval_results

    if args.full:
        args.aopc = True
        args.faithfulness = True
        args.noise_sigma = [0.1, 0.3, 0.6, 1.0]
        args.missing_modality = True

    # ── AOPC Analysis (Section V-F) ───────────────────────────────────
    if args.aopc:
        logging.info("\n" + "=" * 50)
        logging.info("AOPC FAITHFULNESS ANALYSIS")
        logging.info("=" * 50)
        aopc_results = compute_aopc(model, test_loader, device)
        logging.info(f"Baseline WF1: {aopc_results['baseline_wf1']:.2f}%")
        logging.info(f"AOPC Score:   {aopc_results['aopc_score']:.2f}")
        logging.info(f"MCS Ranking:  {aopc_results['rank_order']}")
        for step_name, step_data in aopc_results['steps'].items():
            logging.info(f"  {step_name}: masked={step_data['masked']}, "
                         f"WF1={step_data['wf1']:.2f}%, drop={step_data['drop']:.2f}%")
        all_results['aopc'] = aopc_results

    # ── Faithfulness correlation (Table VI) ────────────────────────────
    if args.faithfulness:
        logging.info("\n" + "=" * 50)
        logging.info("FAITHFULNESS CORRELATION (Table VI)")
        logging.info("=" * 50)
        faith_results = compute_faithfulness(model, test_loader, device)
        logging.info(f"Pearson r = {faith_results['pearson_r']:.2f} "
                     f"(p = {faith_results['pearson_p']:.4f})")
        for r in faith_results['per_modality']:
            logging.info(f"  {r['modality']:8s}: MCS={r['avg_mcs']:.3f}, "
                         f"WF1 drop={r['wf1_drop']:.2f}%")
        all_results['faithfulness'] = faith_results

    # ── Noise sensitivity (Section V-G, Fig. 10) ──────────────────────
    if args.noise_sigma:
        logging.info("\n" + "=" * 50)
        logging.info("NOISE SENSITIVITY ANALYSIS")
        logging.info("=" * 50)
        noise_results = noise_sensitivity_analysis(
            model, test_loader, device, args.noise_sigma
        )
        all_results['noise_sensitivity'] = noise_results

    # ── Missing modality (Table VII) ──────────────────────────────────
    if args.missing_modality:
        logging.info("\n" + "=" * 50)
        logging.info("MISSING MODALITY ROBUSTNESS (Table VII)")
        logging.info("=" * 50)
        missing_results = missing_modality_analysis(model, test_loader, device)
        all_results['missing_modality'] = missing_results

    # ── MCS distribution ──────────────────────────────────────────────
    mcs_dist = mcs_distribution_analysis(model, test_loader, device, args.dataset)
    all_results['mcs_distribution'] = mcs_dist

    # ── Save all results ──────────────────────────────────────────────
    results_path = os.path.join(
        cfg['logging']['log_dir'],
        f"{args.dataset}_eval_results.json"
    )
    # Convert non-serializable types
    serializable = json.loads(json.dumps(all_results, default=lambda x: str(x)))
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logging.info(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()

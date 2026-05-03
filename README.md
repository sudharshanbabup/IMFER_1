# IMFER: Interpretable Multimodal Fusion for Emotion Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"IMFER: An Interpretable Multimodal Fusion Framework for Emotion Recognition in Conversational AI via Cross-Modal Attention and Explainability Mechanisms"**.

---

## Architecture

IMFER comprises four components:

| # | Component | Description | Key Equations |
|---|-----------|-------------|---------------|
| 1 | **Modality Encoders** | RoBERTa (text, 768d), wav2vec 2.0 (audio, 512d), 3D-ResNet (visual, 256d) | Frozen pretrained |
| 2 | **HCMA** | Token-level low-rank cross-modal attention + utterance-level gated fusion | Eq. 1–4 |
| 3 | **CASGT** | Speaker-relationship graph (windowed, O(NW)) + 4-layer transformer | Eq. 5 |
| 4 | **MCS Layer** | Ante-hoc modality attribution via prediction energy decomposition | Eq. 6–8 |

**Training loss** (Eq. 7): `L = L_CE + λ₁·L_MCS + λ₂·L_align` where `λ₁=0.1`, `λ₂=0.05`, `τ=0.07`

---

## Results

| Model | IEMOCAP WF1 | MELD WF1 | EmoryNLP WF1 | #Params |
|-------|:-----------:|:--------:|:------------:|:-------:|
| AIMDiT | 67.34±0.48 | 61.88±0.44 | 39.63 | 86.2M |
| **IMFER** | **69.87±0.21** | **62.34±0.18** | **40.21±0.29** | **59.4M** |

- Paired *t*-test: *p* < 0.01 on IEMOCAP (Cohen's *d* = 2.31)
- ~31% fewer parameters than AIMDiT
- 14.7 ms/utterance inference on Apple M4 Pro

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/<your-username>/IMFER.git
cd IMFER
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# IEMOCAP (requires license from USC)
python scripts/preprocess_iemocap.py --data_dir /path/to/IEMOCAP_full_release --output_dir data/iemocap

# MELD
python scripts/preprocess_meld.py --data_dir /path/to/MELD --output_dir data/meld

# EmoryNLP
python scripts/preprocess_emorynlp.py --data_dir /path/to/EmoryNLP --output_dir data/emorynlp
```

### 3. Train

```bash
# IEMOCAP (5 independent runs)
python train.py --config config/default.yaml --dataset iemocap --data_dir data/iemocap

# MELD
python train.py --config config/default.yaml --dataset meld --data_dir data/meld

# EmoryNLP
python train.py --config config/default.yaml --dataset emorynlp --data_dir data/emorynlp

# Single run (for quick testing)
python train.py --config config/default.yaml --dataset iemocap --data_dir data/iemocap --num_runs 1
```

### 4. Evaluate

```bash
# Standard evaluation (WF1, Acc, MF1, per-class F1, MCS distribution)
python evaluate.py --checkpoint checkpoints/iemocap_run0_best.pt \
    --dataset iemocap --data_dir data/iemocap

# Full analysis (AOPC + faithfulness + noise + missing modality)
python evaluate.py --checkpoint checkpoints/iemocap_run0_best.pt \
    --dataset iemocap --data_dir data/iemocap --full

# AOPC faithfulness only
python evaluate.py --checkpoint checkpoints/iemocap_run0_best.pt \
    --dataset iemocap --data_dir data/iemocap --aopc

# Noise sensitivity (σ = 0.1, 0.3, 0.6, 1.0)
python evaluate.py --checkpoint checkpoints/iemocap_run0_best.pt \
    --dataset iemocap --data_dir data/iemocap --noise_sigma 0.1 0.3 0.6 1.0

# Missing modality robustness (T+A+V, T+A, T+V, T-only)
python evaluate.py --checkpoint checkpoints/iemocap_run0_best.pt \
    --dataset iemocap --data_dir data/iemocap --missing_modality

# Zero-shot cross-dataset transfer (IEMOCAP → MELD)
python evaluate.py --checkpoint checkpoints/iemocap_run0_best.pt \
    --dataset meld --data_dir data/meld --zero_shot
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_k` | 64 | Low-rank projection dimension |
| `d` | 512 | Hidden / fused representation dimension |
| `W` | 10 | CASGT context window |
| `λ₁` | 0.1 | MCS entropy regularization weight |
| `λ₂` | 0.05 | Contrastive alignment weight |
| `τ` | 0.07 | Contrastive temperature |
| Batch size | 32 utterances | — |
| LR (pretrained) | 2×10⁻⁵ | RoBERTa fine-tuning |
| LR (new layers) | 10⁻³ | HCMA, CASGT, MCS |
| Warmup | 10% | Linear warmup |
| Early stopping | patience 10 | On validation WF1 |
| Dropout | 0.3 | — |
| GAT heads | 8 | Graph attention |
| Transformer | 4 layers, 8 heads | CASGT encoder |

---

## Project Structure

```
IMFER/
├── config/
│   └── default.yaml              # All hyperparameters (Section IV-C)
├── imfer/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py           # RoBERTa, wav2vec2, 3D-ResNet (Section III-C)
│   │   ├── hcma.py               # Hierarchical Cross-Modal Attention (Section III-D)
│   │   ├── casgt.py              # Context-Aware Speaker Graph Transformer (Section III-E)
│   │   ├── mcs.py                # Modality Contribution Score layer (Section III-F)
│   │   └── imfer.py              # Full IMFER model + loss (Eq. 7-10)
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py            # Dataset classes + collation (Section IV-A)
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # WF1, Acc, MF1, AOPC, faithfulness (Section V)
│       └── helpers.py            # Seed, device, EarlyStopping
├── scripts/
│   ├── preprocess_iemocap.py     # IEMOCAP feature extraction
│   ├── preprocess_meld.py        # MELD feature extraction
│   └── preprocess_emorynlp.py    # EmoryNLP feature extraction
├── train.py                      # Training loop (5 runs, dual LR, warmup)
├── evaluate.py                   # Evaluation + all analyses
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Hardware

All experiments were conducted on a single **Apple M4 Pro** (12-core CPU, 20-core GPU, 48 GB unified memory) using PyTorch MPS backend. The code also supports CUDA GPUs and CPU.

---

## Citation

```bibtex
@article{sathalla2025imfer,
  title={{IMFER}: An Interpretable Multimodal Fusion Framework for Emotion Recognition
         in Conversational {AI} via Cross-Modal Attention and Explainability Mechanisms},
  author={Sathalla, Suresh and Babu, Pandava Sudharshan and Ramegowda, Mahesh and Gottam, Omprakash},
  journal={IEEE Transactions on Affective Computing},
  year={2025}
}
```

## Acknowledgement

The architectural overview figure was generated with the assistance of Claude (Anthropic). AI-assisted tools were also used for grammar checking and proofreading of the manuscript. The authors retain full responsibility for all scientific content.

## License

MIT License — see [LICENSE](LICENSE).

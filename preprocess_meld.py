"""
MELD Preprocessing — Section IV-A.

Dataset: Naturalistic TV-show dialogue (Friends), 13708 utterances,
         7 emotion classes, multiple speakers per episode.
Classes: neutral, surprise, fear, sadness, joy, disgust, anger

Expected directory structure:
    /path/to/MELD/
        train_sent_emo.csv
        dev_sent_emo.csv
        test_sent_emo.csv
        train/audio/     (optional: wav files per utterance)
        dev/audio/
        test/audio/
        train/video/     (optional: mp4 files per utterance)
        dev/video/
        test/video/

Usage:
    python scripts/preprocess_meld.py --data_dir /path/to/MELD --output_dir data/meld
"""

import os
import json
import argparse
import csv

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel


LABEL_MAP = {
    "neutral": 0, "surprise": 1, "fear": 2, "sadness": 3,
    "joy": 4, "disgust": 5, "anger": 6,
}


def load_meld_csv(csv_path):
    """Parse MELD CSV file into conversations."""
    conversations = {}

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dia_id = int(row.get("Dialogue_ID", 0))
                utt_id = int(row.get("Utterance_ID", 0))
                text = row.get("Utterance", "")
                emotion = row.get("Emotion", "").lower().strip()
                speaker = row.get("Speaker", "unknown")

                if emotion not in LABEL_MAP:
                    continue

                if dia_id not in conversations:
                    conversations[dia_id] = []

                conversations[dia_id].append({
                    'utt_id': utt_id,
                    'text': text,
                    'emotion': LABEL_MAP[emotion],
                    'speaker': speaker,
                })
            except (ValueError, KeyError):
                continue

    # Sort utterances within each conversation
    for dia_id in conversations:
        conversations[dia_id].sort(key=lambda x: x['utt_id'])

    return conversations


def extract_text_features_batch(texts, tokenizer, model, device, max_len=128):
    """Extract RoBERTa features for a batch of texts."""
    features, masks = [], []

    for text in texts:
        enc = tokenizer(
            text, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            feat = out.last_hidden_state.squeeze(0).cpu()
        features.append(feat)
        masks.append(enc["attention_mask"].squeeze(0).bool())

    return torch.stack(features), torch.stack(masks)


def main():
    parser = argparse.ArgumentParser(description="Preprocess MELD")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/meld")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.join(args.output_dir, "conversations"), exist_ok=True)

    print("Loading RoBERTa...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    text_model = RobertaModel.from_pretrained("roberta-base").to(device).eval()

    split_files = {
        "train": "train_sent_emo.csv",
        "val": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }

    all_splits = {}
    conv_count = 0

    for split, filename in split_files.items():
        csv_path = os.path.join(args.data_dir, filename)
        if not os.path.exists(csv_path):
            print(f"  {csv_path} not found, skipping {split}")
            continue

        conversations = load_meld_csv(csv_path)
        split_ids = []

        for dia_id, utts in sorted(conversations.items()):
            texts = [u['text'] for u in utts]
            labels = [u['emotion'] for u in utts]

            # Speaker mapping within conversation
            speaker_set = sorted(set(u['speaker'] for u in utts))
            speaker_to_idx = {s: i for i, s in enumerate(speaker_set)}
            speakers = [speaker_to_idx[u['speaker']] for u in utts]

            if len(texts) == 0:
                continue

            # Text features
            text_feats, text_masks = extract_text_features_batch(
                texts, tokenizer, text_model, device
            )

            N = len(texts)
            # Audio/visual placeholders (replace with actual extraction)
            audio_feats = torch.zeros(N, 50, 512)
            visual_feats = torch.zeros(N, 1, 256)

            conv_data = {
                'text_features': text_feats,
                'audio_features': audio_feats,
                'visual_features': visual_feats,
                'text_mask': text_masks,
                'speaker_ids': torch.tensor(speakers, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'num_utterances': N,
            }

            conv_name = f"conv_{conv_count:04d}"
            torch.save(conv_data, os.path.join(
                args.output_dir, "conversations", f"{conv_name}.pt"
            ))
            split_ids.append(conv_name)
            conv_count += 1

        all_splits[split] = split_ids
        print(f"  {split}: {len(split_ids)} conversations")

    with open(os.path.join(args.output_dir, "splits.json"), "w") as f:
        json.dump(all_splits, f, indent=2)

    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"\nDone! {conv_count} conversations total.")


if __name__ == "__main__":
    main()

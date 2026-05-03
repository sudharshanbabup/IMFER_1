"""
EmoryNLP Preprocessing — Section IV-A.

Dataset: Text-centric scripted TV-show, 12606 utterances,
         7 labels aligned with Willcox's emotion wheel.
Classes: joyful, mad, peaceful, powerful, sad, scared, neutral

Expected directory structure:
    /path/to/EmoryNLP/
        emorynlp_train_final.csv  (or JSON format)
        emorynlp_dev_final.csv
        emorynlp_test_final.csv

Usage:
    python scripts/preprocess_emorynlp.py --data_dir /path/to/EmoryNLP --output_dir data/emorynlp
"""

import os
import json
import argparse
import csv

import torch
from transformers import RobertaTokenizer, RobertaModel


LABEL_MAP = {
    "joyful": 0, "mad": 1, "peaceful": 2, "powerful": 3,
    "sad": 4, "scared": 5, "neutral": 6,
}


def load_emorynlp_csv(csv_path):
    """Parse EmoryNLP CSV into conversations."""
    conversations = {}

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # EmoryNLP uses Season_Episode_Scene as dialogue ID
                season = row.get("Season", "0")
                episode = row.get("Episode", "0")
                scene = row.get("Scene_ID", row.get("Scene", "0"))
                dia_id = f"{season}_{episode}_{scene}"

                utt_id = int(row.get("Utterance_ID", row.get("Utterance_ID", 0)))
                text = row.get("Utterance", row.get("Transcript", ""))
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

    for dia_id in conversations:
        conversations[dia_id].sort(key=lambda x: x['utt_id'])

    return conversations


def load_emorynlp_json(json_path):
    """Parse EmoryNLP JSON format (alternative)."""
    conversations = {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "episodes" in data:
        episodes = data["episodes"]
    elif isinstance(data, list):
        episodes = data
    else:
        return conversations

    conv_idx = 0
    for ep in episodes:
        scenes = ep.get("scenes", [ep]) if isinstance(ep, dict) else [ep]
        for scene in scenes:
            utts = scene.get("utterances", scene.get("turns", []))
            conv_data = []
            for i, utt in enumerate(utts):
                text = utt.get("transcript", utt.get("text", ""))
                emotion = utt.get("emotion", "").lower().strip()
                speaker = utt.get("speaker", utt.get("speakers", ["unknown"])[0]
                                  if isinstance(utt.get("speakers"), list) else "unknown")

                if emotion in LABEL_MAP:
                    conv_data.append({
                        'utt_id': i,
                        'text': text,
                        'emotion': LABEL_MAP[emotion],
                        'speaker': speaker if isinstance(speaker, str) else str(speaker),
                    })

            if conv_data:
                conversations[f"conv_{conv_idx}"] = conv_data
                conv_idx += 1

    return conversations


def extract_text_features(texts, tokenizer, model, device, max_len=128):
    """Extract RoBERTa features."""
    features, masks = [], []
    for text in texts:
        enc = tokenizer(text, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
            feat = out.last_hidden_state.squeeze(0).cpu()
        features.append(feat)
        masks.append(enc["attention_mask"].squeeze(0).bool())
    return torch.stack(features), torch.stack(masks)


def main():
    parser = argparse.ArgumentParser(description="Preprocess EmoryNLP")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/emorynlp")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.join(args.output_dir, "conversations"), exist_ok=True)

    print("Loading RoBERTa...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    text_model = RobertaModel.from_pretrained("roberta-base").to(device).eval()

    # Try CSV first, then JSON
    split_files = {}
    for split, prefixes in [
        ("train", ["emorynlp_train_final", "train"]),
        ("val", ["emorynlp_dev_final", "dev"]),
        ("test", ["emorynlp_test_final", "test"]),
    ]:
        for prefix in prefixes:
            for ext in [".csv", ".json"]:
                path = os.path.join(args.data_dir, prefix + ext)
                if os.path.exists(path):
                    split_files[split] = path
                    break
            if split in split_files:
                break

    all_splits = {}
    conv_count = 0

    for split, filepath in split_files.items():
        if filepath.endswith(".json"):
            conversations = load_emorynlp_json(filepath)
        else:
            conversations = load_emorynlp_csv(filepath)

        split_ids = []

        for dia_id, utts in sorted(conversations.items()):
            texts = [u['text'] for u in utts]
            labels = [u['emotion'] for u in utts]

            speaker_set = sorted(set(u['speaker'] for u in utts))
            speaker_to_idx = {s: i for i, s in enumerate(speaker_set)}
            speakers = [speaker_to_idx[u['speaker']] for u in utts]

            if not texts:
                continue

            text_feats, text_masks = extract_text_features(
                texts, tokenizer, text_model, device
            )

            N = len(texts)
            # EmoryNLP is text-centric; audio/visual are placeholders
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

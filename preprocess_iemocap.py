"""
IEMOCAP Preprocessing — Section IV-A.

Dataset: 10 speakers, scripted + improvised, 5531 utterances
Classes: happy, sad, neutral, angry, excited, frustrated (6 classes)

Preprocessing (Section IV-C):
    - Text: tokenised with RoBERTa BPE tokeniser (max 128 tokens)
    - Audio: resampled to 16 kHz; wav2vec 2.0 features from wav2vec2-base-960h
    - Video: frames at 30 fps; 3D-ResNet on 16-frame clips centred on utterance

Output: data/iemocap/conversations/conv_XXXX.pt per conversation
    {
        'text_features':   (N, 128, 768)
        'audio_features':  (N, T_a, 512)
        'visual_features': (N, T_v, 256)
        'text_mask':       (N, 128)
        'speaker_ids':     (N,)
        'labels':          (N,)
        'num_utterances':  N
    }

Usage:
    python scripts/preprocess_iemocap.py \
        --data_dir /path/to/IEMOCAP_full_release \
        --output_dir data/iemocap
"""

import os
import sys
import json
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model

# ── IEMOCAP label mapping ─────────────────────────────────────────────────

LABEL_MAP = {
    "hap": 0, "happy": 0,
    "sad": 1, "sadness": 1,
    "neu": 2, "neutral": 2,
    "ang": 3, "angry": 3, "anger": 3,
    "exc": 4, "excited": 4, "excitement": 4,
    "fru": 5, "frustrated": 5, "frustration": 5,
}
VALID_LABELS = {0, 1, 2, 3, 4, 5}


def parse_iemocap_annotations(session_dir):
    """Parse IEMOCAP .txt annotation files into utterance metadata."""
    emo_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
    utterances = {}

    for emo_file in sorted(glob.glob(os.path.join(emo_dir, "*.txt"))):
        with open(emo_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                # Format: [start - end] utt_id emotion [V, A, D]
                match = re.match(
                    r'\[(\d+\.\d+) - (\d+\.\d+)\]\s+(\S+)\s+(\S+)\s+\[.*\]',
                    line
                )
                if match:
                    start_t = float(match.group(1))
                    end_t = float(match.group(2))
                    utt_id = match.group(3)
                    emotion = match.group(4).lower()

                    if emotion in LABEL_MAP and LABEL_MAP[emotion] in VALID_LABELS:
                        utterances[utt_id] = {
                            'start': start_t,
                            'end': end_t,
                            'emotion': LABEL_MAP[emotion],
                            'speaker': utt_id.split("_")[-1][0],  # M or F
                        }

    return utterances


def extract_text_features(texts, tokenizer, model, device, max_len=128):
    """Extract RoBERTa features for a list of texts."""
    features = []
    masks = []

    for text in texts:
        enc = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            feat = out.last_hidden_state.squeeze(0).cpu()  # (max_len, 768)

        features.append(feat)
        masks.append(attention_mask.squeeze(0).cpu().bool())

    return torch.stack(features), torch.stack(masks)


def extract_audio_features(audio_path, start, end, processor, model, device,
                           target_sr=16000, proj=None):
    """Extract wav2vec 2.0 features for an utterance segment."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Extract segment
    start_sample = int(start * target_sr)
    end_sample = int(end * target_sr)
    segment = waveform[:, start_sample:end_sample]

    if segment.size(1) == 0:
        segment = torch.zeros(1, target_sr)  # 1-second silence fallback

    # Mono
    if segment.size(0) > 1:
        segment = segment.mean(dim=0, keepdim=True)
    segment = segment.squeeze(0)

    # wav2vec2 processing
    inputs = processor(segment.numpy(), sampling_rate=target_sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        out = model(input_values=input_values)
        hidden = out.last_hidden_state.squeeze(0).cpu()  # (T_a, 768)

    # Project to 512 if projection provided
    if proj is not None:
        with torch.no_grad():
            hidden = proj(hidden.to(device)).cpu()

    return hidden


def extract_video_features(video_dir, utt_id, start, end, visual_model, device,
                           fps=30, clip_frames=16):
    """
    Extract 3D-ResNet features for a 16-frame clip centred on the utterance.
    Falls back to zeros if video is unavailable.
    """
    # In practice, you would extract frames from the video file here.
    # IEMOCAP provides .avi files; use OpenCV or moviepy to extract frames.
    # For this script, we create a placeholder if video processing is not set up.

    try:
        import cv2
        video_path = os.path.join(video_dir, f"{utt_id}.avi")
        if not os.path.exists(video_path):
            # Try session-level video
            parts = utt_id.rsplit("_", 1)
            dialog_id = parts[0] if len(parts) > 1 else utt_id
            video_path = os.path.join(video_dir, f"{dialog_id}.avi")

        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            mid_time = (start + end) / 2
            start_frame = max(0, int((mid_time - clip_frames / (2 * vid_fps)) * vid_fps))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for _ in range(clip_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame).float() / 255.0)
            cap.release()

            while len(frames) < clip_frames:
                frames.append(torch.zeros(112, 112, 3))

            clip = torch.stack(frames[:clip_frames])  # (T, H, W, C)
            clip = clip.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)

            with torch.no_grad():
                feat = visual_model(clip.to(device)).squeeze(0).cpu()
            return feat
    except ImportError:
        pass

    # Fallback: zero features with T_v=1
    return torch.zeros(1, 256)


def main():
    parser = argparse.ArgumentParser(description="Preprocess IEMOCAP")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to IEMOCAP_full_release/")
    parser.add_argument("--output_dir", type=str, default="data/iemocap")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_video", action="store_true",
                        help="Skip video feature extraction (use zeros)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.join(args.output_dir, "conversations"), exist_ok=True)

    print("Loading models...")

    # Text encoder
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    text_model = RobertaModel.from_pretrained("roberta-base").to(device).eval()

    # Audio encoder
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    audio_proj = torch.nn.Linear(768, 512).to(device)
    torch.nn.init.xavier_uniform_(audio_proj.weight)

    # Visual encoder (if not skipping)
    if not args.skip_video:
        from imfer.models.encoders import VisualEncoder
        visual_model = VisualEncoder(out_dim=256, freeze=True).to(device).eval()
    else:
        visual_model = None

    print("Processing sessions...")

    speaker_map = {}
    train_ids, val_ids, test_ids = [], [], []
    conv_count = 0

    for session_idx in range(1, 6):
        session_dir = os.path.join(args.data_dir, f"Session{session_idx}")
        if not os.path.isdir(session_dir):
            print(f"  Session {session_idx} not found, skipping")
            continue

        utterances = parse_iemocap_annotations(session_dir)
        if not utterances:
            continue

        # Group utterances by dialog
        dialogs = {}
        for utt_id, meta in utterances.items():
            # Dialog ID: everything before the last underscore + number
            parts = utt_id.rsplit("_", 1)
            dialog_id = parts[0]
            if dialog_id not in dialogs:
                dialogs[dialog_id] = []
            dialogs[dialog_id].append((utt_id, meta))

        for dialog_id, utt_list in sorted(dialogs.items()):
            utt_list.sort(key=lambda x: x[1]['start'])

            texts = []
            audio_feats = []
            visual_feats = []
            speakers = []
            labels = []

            # Load transcript
            transcript_dir = os.path.join(session_dir, "dialog", "transcriptions")
            transcript_file = os.path.join(transcript_dir, f"{dialog_id}.txt")
            utt_texts = {}

            if os.path.exists(transcript_file):
                with open(transcript_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        match = re.match(r'(\S+)\s+\[.*\]:\s+(.*)', line.strip())
                        if match:
                            utt_texts[match.group(1)] = match.group(2)

            for utt_id, meta in utt_list:
                text = utt_texts.get(utt_id, "")
                if not text:
                    text = "[empty]"
                texts.append(text)

                # Speaker mapping
                spk = f"S{session_idx}_{meta['speaker']}"
                if spk not in speaker_map:
                    speaker_map[spk] = len(speaker_map)
                speakers.append(speaker_map[spk])

                labels.append(meta['emotion'])

                # Audio features
                wav_dir = os.path.join(session_dir, "sentences", "wav", dialog_id)
                wav_path = os.path.join(wav_dir, f"{utt_id}.wav")
                if os.path.exists(wav_path):
                    a_feat = extract_audio_features(
                        wav_path, 0, meta['end'] - meta['start'],
                        wav2vec_processor, wav2vec_model, device, proj=audio_proj
                    )
                else:
                    a_feat = torch.zeros(1, 512)
                audio_feats.append(a_feat)

                # Visual features
                if not args.skip_video and visual_model is not None:
                    video_dir = os.path.join(session_dir, "dialog", "avi", "DivX")
                    v_feat = extract_video_features(
                        video_dir, utt_id, meta['start'], meta['end'],
                        visual_model, device
                    )
                else:
                    v_feat = torch.zeros(1, 256)
                visual_feats.append(v_feat)

            if len(texts) == 0:
                continue

            # Extract text features
            text_feats, text_masks = extract_text_features(
                texts, tokenizer, text_model, device
            )

            # Pad audio and visual to consistent lengths
            max_audio_len = max(a.size(0) for a in audio_feats)
            max_visual_len = max(v.size(0) for v in visual_feats)

            audio_padded = torch.zeros(len(audio_feats), max_audio_len, 512)
            visual_padded = torch.zeros(len(visual_feats), max_visual_len, 256)
            for i, a in enumerate(audio_feats):
                audio_padded[i, :a.size(0)] = a
            for i, v in enumerate(visual_feats):
                visual_padded[i, :v.size(0)] = v

            conv_data = {
                'text_features': text_feats,          # (N, 128, 768)
                'audio_features': audio_padded,       # (N, T_a, 512)
                'visual_features': visual_padded,     # (N, T_v, 256)
                'text_mask': text_masks,              # (N, 128)
                'speaker_ids': torch.tensor(speakers, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'num_utterances': len(texts),
            }

            conv_name = f"conv_{conv_count:04d}"
            torch.save(conv_data, os.path.join(
                args.output_dir, "conversations", f"{conv_name}.pt"
            ))

            # Split: Session 5 = test, Session 4 = val, rest = train
            if session_idx == 5:
                test_ids.append(conv_name)
            elif session_idx == 4:
                val_ids.append(conv_name)
            else:
                train_ids.append(conv_name)

            conv_count += 1

        print(f"  Session {session_idx}: {len(dialogs)} dialogs processed")

    # Save splits
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(os.path.join(args.output_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    # Save label map
    label_map = {
        "happy": 0, "sad": 1, "neutral": 2,
        "angry": 3, "excited": 4, "frustrated": 5,
    }
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    total = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"\nDone! {total} conversations saved.")
    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Alternate Step 4 -- build a Whisper fine-tuning dataset from pre-segmented
Hugging Face datasets (e.g. the MohammadGholizadeh Persian YouTube/podcast
collection) instead of from Speechmatics output. Produces the same on-disk
shape sm_04_build_dataset.py does (HF DatasetDict of {audio: filename, text}
plus a sibling clips/ folder), so train_whisper.py needs no changes.

These repos already ship one row per utterance with wav audio and a text
column -- there is nothing to segment. What differs per source is the text
column name (transcription vs sentence) and whether a validation split
already exists (only youtube-farsi has one; the rest are a single "train").

Streams rather than downloads: several of these repos are 100+ GB of parquet
if pulled whole. --max-seconds-per-repo stops requesting shards once that
repo's budget is met -- with it unset you will pull the ENTIRE repo to disk
before this script gets to filter anything, which for the largest repo in
this collection is ~132GB.

A repo with no video/episode id column and only a "train" split (i.e. every
podcast/channel repo here) is, for leakage purposes, ONE recording -- there
is no source column to split on the way sm_04 splits by video_id. The
fallback is a tail slice of --val-fraction, mirroring the single-video
fallback in sm_04_build_dataset.py. It is not as clean as a held-out
recording, but it is what's available without re-diarizing the source.

Usage:
    python hf_build_dataset.py \
        --repo MohammadGholizadeh/youtube-farsi:transcription \
        --repo MohammadGholizadeh/channelbpodcast_dataset_persian:sentence \
        --out ./sm_data/collection --max-seconds-per-repo 18000

Requires: datasets, soundfile, plus ffmpeg on PATH for any source whose audio
container soundfile can't decode directly.
"""
import argparse
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_dataset

from sm_04_build_dataset import normalize_fa

SAMPLE_RATE = 16000


def decode_row_audio(raw: dict):
    """raw is {"bytes": ..., "path": ...} from an Audio(decode=False) column --
    decode=False so this never touches the Audio feature's torchcodec-backed
    decoder (see the same avoidance in train_whisper.prepare). soundfile
    handles the wav/flac case directly; anything else falls back to ffmpeg,
    same tool sm_04 already requires on this box."""
    try:
        audio, sr = sf.read(io.BytesIO(raw["bytes"]), dtype="float32")
    except Exception:
        suffix = Path(raw.get("path") or "clip.audio").suffix or ".audio"
        with tempfile.NamedTemporaryFile(suffix=suffix) as src, \
             tempfile.NamedTemporaryFile(suffix=".wav") as dst:
            src.write(raw["bytes"])
            src.flush()
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src.name,
                 "-ar", str(SAMPLE_RATE), "-ac", "1", dst.name],
                check=True,
            )
            audio, sr = sf.read(dst.name, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def resample_if_needed(audio, sr):
    if sr == SAMPLE_RATE:
        return audio
    with tempfile.NamedTemporaryFile(suffix=".wav") as src, \
         tempfile.NamedTemporaryFile(suffix=".wav") as dst:
        sf.write(src.name, audio, sr)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src.name,
             "-ar", str(SAMPLE_RATE), "-ac", "1", dst.name],
            check=True,
        )
        audio, _ = sf.read(dst.name, dtype="float32")
    return audio


def parse_repo_spec(spec: str):
    """repo[:text_column[:split]] -- text_column defaults to 'sentence'
    (what most of this collection uses), split defaults to 'train'."""
    parts = spec.split(":")
    repo = parts[0]
    text_col = parts[1] if len(parts) > 1 else "sentence"
    split = parts[2] if len(parts) > 2 else "train"
    return repo, text_col, split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", action="append", required=True, dest="repos",
        help="repo[:text_column[:split]], repeatable. E.g. "
        "MohammadGholizadeh/youtube-farsi:transcription:train",
    )
    parser.add_argument("--out", default="./sm_data/collection")
    parser.add_argument(
        "--max-seconds-per-repo", type=float, default=None,
        help="stop pulling a repo once this many seconds of KEPT audio are "
        "collected. Unset means take the entire repo -- fine for "
        "channelbpodcast_dataset_persian (1.2GB), NOT for bplus_podcast_persian "
        "or Tabaghe16_dataset_persian (90-130GB each)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="for repos with only a train split, hold out this fraction as a "
        "TAIL slice of parquet row order (approximates a held-out recording; "
        "see module docstring for why it is only approximate here)",
    )
    parser.add_argument("--min-sec", type=float, default=0.3)
    parser.add_argument("--max-sec", type=float, default=28.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = [], []
    for spec in args.repos:
        repo, text_col, split = parse_repo_spec(spec)
        tag = repo.split("/")[-1]
        print(f"[{tag}] streaming split={split} (text_col={text_col}) ...", file=sys.stderr)
        ds = load_dataset(repo, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))

        has_named_val = split in ("validation", "val", "test")
        rows, total_sec = [], 0.0
        for i, row in enumerate(ds):
            if args.max_seconds_per_repo and total_sec >= args.max_seconds_per_repo:
                break
            text = normalize_fa(row.get(text_col) or "")
            if not text:
                continue
            audio, sr = decode_row_audio(row["audio"])
            audio = resample_if_needed(audio, sr)
            dur = len(audio) / SAMPLE_RATE
            if dur < args.min_sec or dur > args.max_sec:
                continue
            name = f"{tag}_{i:06d}.wav"
            sf.write(clips_dir / name, audio, SAMPLE_RATE)
            rows.append({"audio": name, "text": text, "source": tag, "duration": round(dur, 3)})
            total_sec += dur
            if len(rows) % 500 == 0:
                print(f"[{tag}] {len(rows)} clips, {total_sec / 3600:.2f}h", file=sys.stderr)

        print(f"[{tag}] done: {len(rows)} clips, {total_sec / 3600:.2f}h "
              f"({'named val/test split' if has_named_val else 'train, tail-sliced for val'})",
              file=sys.stderr)

        if has_named_val:
            val_rows.extend(rows)
        else:
            cut = max(1, int(len(rows) * args.val_fraction)) if rows else 0
            val_rows.extend(rows[-cut:] if cut else [])
            train_rows.extend(rows[:-cut] if cut else rows)

    if not train_rows:
        sys.exit("no training clips produced")
    if not val_rows:
        sys.exit("no validation clips produced -- pass a repo with a val/test split "
                 "or raise --val-fraction")

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows + val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    DatasetDict({
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(val_rows),
    }).save_to_disk(str(out_dir / "hf_dataset"))

    total_h = sum(r["duration"] for r in train_rows + val_rows) / 3600
    print(f"\ntotal: {len(train_rows) + len(val_rows)} clips, {total_h:.2f}h "
          f"(train={len(train_rows)}, val={len(val_rows)})")
    print(f"dataset -> {out_dir / 'hf_dataset'}")
    print(f"clips   -> {clips_dir}")


if __name__ == "__main__":
    main()

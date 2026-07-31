#!/usr/bin/env python3
"""
Step 4b -- push the local Whisper dataset (built by sm_04_build_dataset.py from
sm_data/audio + sm_data/transcripts) to the Hugging Face Hub.

sm_04_build_dataset.py's on-disk shape is a DatasetDict whose "audio" column
holds bare clip filenames, with the actual wav files in a sibling clips/
folder -- not something push_to_hub can upload directly. This resolves each
filename to its real path and casts the column to an Audio feature so the
clip bytes get embedded into the pushed parquet shards, the same layout
MohammadGholizadeh/Tabaghe16_dataset_persian and friends use.

One-time setup:
    pip install huggingface_hub
    huggingface-cli login          # needs a token with write access

Usage:
    python sm_04_build_dataset.py --audio ./sm_data/audio \
        --transcripts ./sm_data/transcripts --out ./sm_data/dataset
    python push_to_hub.py --dataset ./sm_data/dataset --repo shervingh2000/behpardaz
"""
import argparse
from pathlib import Path

from datasets import Audio, load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./sm_data/dataset/hf_dataset",
                        help="local DatasetDict built by sm_04_build_dataset.py "
                        "(that script writes it to <out>/hf_dataset, with clips as a "
                        "sibling <out>/clips -- NOT a child of the dataset dir)")
    parser.add_argument("--clips-dir", default=None,
                        help="dir holding the audio clips referenced by --dataset; "
                        "defaults to a 'clips' folder next to (not inside) it, matching "
                        "sm_04_build_dataset.py's layout")
    parser.add_argument("--repo", default="shervingh2000/behpardaz")
    parser.add_argument("--private", action="store_true",
                        help="push as a private dataset repo")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    clips_dir = Path(args.clips_dir) if args.clips_dir else dataset_dir.parent / "clips"
    if not clips_dir.is_dir():
        raise SystemExit(f"clips dir not found: {clips_dir.resolve()} (pass --clips-dir)")

    ds = load_from_disk(str(dataset_dir))
    print(ds)

    ds = ds.map(lambda r: {"audio": str(clips_dir / r["audio"])})
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    ds.push_to_hub(args.repo, private=args.private)
    print(f"pushed -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

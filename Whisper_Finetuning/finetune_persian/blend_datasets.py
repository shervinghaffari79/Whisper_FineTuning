#!/usr/bin/env python3
"""
Step 4c -- blend several built datasets into one, so stage-2 training can add
broad corpora WITHOUT dropping the original domain, and so the validation set
can grow by appending new voices to the ones already there.

Consumes and produces the same on-disk shape as sm_04_build_dataset.py and
hf_build_dataset.py (an HF DatasetDict of {audio: filename, text: str} plus a
sibling clips/ folder), so train_whisper.py and eval_baseline.py read the
output with no changes.

Why blend rather than just train on the new corpus: a LoRA adapter is ~15.7M
trainable params. Several thousand steps of differently-styled audio can
dilute a narrow-domain specialisation even while aggregate WER looks fine.
Keeping the original clips in the mix -- oversampled if they are the smaller
set -- is what holds that ground.

    --train PATH[:take]   repeatable; uses PATH's "train" split
    --val   PATH[:take]   repeatable; uses PATH's "validation" split

take controls how much of a source is used:
    12000   cap at 12000 clips (shuffled with --seed, so it is a real sample
            and not the opening clips of whichever recording sorts first)
    0.25    take 25% of the split
    x3      repeat every row 3 times (oversample the small in-domain set so it
            keeps weight against a much larger corpus; costs no extra disk,
            the rows point at the same clip files)
    omitted take the split whole

EVERY row carries a "source" column naming which dataset it came from.
eval_baseline.py breaks WER down by that column, which is the only thing that
keeps a grown validation set honest: appending new voices changes the
population, so a single blended WER is NOT comparable to a number measured
before the append. The per-source rows still are.

Clips are hard-linked into the merged folder when the filesystem allows it
(same NTFS volume) and copied otherwise -- a blended corpus can be hundreds of
GB and duplicating it is usually not affordable.

Usage:
    python blend_datasets.py \
        --train ./sm_data/dataset/hf_dataset:x3 \
        --train ./sm_data/collection/hf_dataset:20000 \
        --val   ./sm_data/dataset/hf_dataset \
        --val   ./sm_data/collection/hf_dataset:400 \
        --out   ./sm_data/blend
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


def parse_spec(spec: str):
    """PATH[:take] -- rsplit on ':' so Windows drive letters (C:\\...) survive."""
    if ":" in spec:
        head, tail = spec.rsplit(":", 1)
        # a bare drive letter means there was no take suffix at all
        if head and tail and not (len(head) == 1 and head.isalpha()):
            return head, tail
    return spec, None


def resolve_take(take, n):
    """-> (count, repeat). count caps rows, repeat duplicates them."""
    if take is None:
        return n, 1
    if take.startswith("x"):
        return n, int(take[1:])
    value = float(take)
    if value <= 1.0 and "." in take:
        return max(1, int(round(n * value))), 1
    return min(int(value), n), 1


def link_or_copy(src: Path, dest: Path, stats: dict):
    if dest.exists():
        return
    try:
        os.link(src, dest)
        stats["linked"] += 1
    except OSError:
        shutil.copy2(src, dest)
        stats["copied"] += 1


def collect(specs, split, clips_out, seed, stats):
    """Load each spec's split, copy its clips into the merged folder under a
    source-prefixed name, and return the rewritten rows."""
    parts = []
    for spec in specs:
        path, take = parse_spec(spec)
        ds_path = Path(path)
        clips_in = ds_path.parent / "clips"
        if not clips_in.is_dir():
            sys.exit(f"clips dir not found for {ds_path}: {clips_in}\n"
                     f"(expected a 'clips' folder beside the hf_dataset dir)")
        dd = load_from_disk(str(ds_path))
        if split not in dd:
            sys.exit(f"{ds_path} has no '{split}' split (has: {list(dd)})")
        ds = dd[split]
        # tag by the dataset ROOT dir (sm_data/dataset, sm_data/collection),
        # not the hf_dataset leaf, which is the same word for every source
        tag = ds_path.parent.name

        count, repeat = resolve_take(take, len(ds))
        if count < len(ds):
            ds = ds.shuffle(seed=seed).select(range(count))

        rows = []
        for ex in ds:
            name = f"{tag}__{ex['audio']}"
            link_or_copy(clips_in / ex["audio"], clips_out / name, stats)
            rows.append({"audio": name, "text": ex["text"], "source": tag})
        if repeat > 1:
            rows = rows * repeat

        print(f"  [{split}] {tag}: {len(ds)} clips"
              + (f" x{repeat} = {len(rows)} rows" if repeat > 1 else ""),
              file=sys.stderr)
        parts.append(Dataset.from_list(rows))
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", required=True, dest="trains",
                        metavar="PATH[:take]", help="hf_dataset dir; uses its train split")
    parser.add_argument("--val", action="append", required=True, dest="vals",
                        metavar="PATH[:take]", help="hf_dataset dir; uses its validation split")
    parser.add_argument("--out", default="./sm_data/blend")
    parser.add_argument("--seed", type=int, default=42,
                        help="selects which clips a numeric/fractional take draws")
    parser.add_argument("--shuffle-train", action="store_true", default=True,
                        help="interleave sources in the train split (default). Without it "
                        "the trainer sees one whole corpus then the next, and the last "
                        "corpus dominates the final -- and therefore selected -- weights")
    args = parser.parse_args()

    out_dir = Path(args.out)
    clips_out = out_dir / "clips"
    clips_out.mkdir(parents=True, exist_ok=True)

    stats = {"linked": 0, "copied": 0}
    print("collecting train sources...", file=sys.stderr)
    train = collect(args.trains, "train", clips_out, args.seed, stats)
    print("collecting validation sources...", file=sys.stderr)
    val = collect(args.vals, "validation", clips_out, args.seed, stats)

    if args.shuffle_train:
        train = train.shuffle(seed=args.seed)

    overlap = set(train["audio"]) & set(val["audio"])
    if overlap:
        sys.exit(f"{len(overlap)} clips appear in BOTH train and validation, e.g. "
                 f"{sorted(overlap)[:3]}\nthat leaks the eval set into training -- check "
                 f"that no --train and --val spec name the same dataset AND split")

    DatasetDict({"train": train, "validation": val}).save_to_disk(str(out_dir / "hf_dataset"))

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for split, ds in (("train", train), ("validation", val)):
            for ex in ds:
                f.write(json.dumps({**ex, "split": split}, ensure_ascii=False) + "\n")

    def by_source(ds):
        counts = {}
        for s in ds["source"]:
            counts[s] = counts.get(s, 0) + 1
        return counts

    print(f"\ntrain      : {len(train)} rows  {by_source(train)}")
    print(f"validation : {len(val)} rows  {by_source(val)}")
    print(f"clips      : {stats['linked']} hard-linked, {stats['copied']} copied")
    print(f"dataset -> {out_dir / 'hf_dataset'}")
    print(f"clips   -> {clips_out}")


if __name__ == "__main__":
    main()

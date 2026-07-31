#!/usr/bin/env python3
"""
Build a Whisper fine-tuning dataset from pre-segmented Hugging Face datasets
(e.g. the MohammadGholizadeh Persian YouTube/podcast collection, or your own
pushed dataset). Produces an HF DatasetDict of {audio: filename, text} plus a
sibling clips/ folder, the shape train_whisper.py and push_to_hub.py expect.

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
is no source column to split on. The fallback is a tail slice of
--val-fraction. It is not as clean as a held-out recording, but it is what's
available without re-diarizing the source.

Usage:
    python hf_build_dataset.py \
        --repo MohammadGholizadeh/youtube-farsi:transcription \
        --repo MohammadGholizadeh/channelbpodcast_dataset_persian:sentence \
        --out ./sm_data/collection --max-seconds-per-repo 18000

Requires: datasets, soundfile, plus ffmpeg on PATH for any source whose audio
container soundfile can't decode directly.

For a private repo, set HF_TOKEN in the environment before running -- a repo
with no video/episode/id column also needs --id-column matching whatever
column names the recording.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_dataset

from fa_text import normalize_fa

SAMPLE_RATE = 16000


def decode_row_audio(raw: dict):
    """raw is {"bytes": ..., "path": ...} from an Audio(decode=False) column --
    decode=False so this never touches the Audio feature's torchcodec-backed
    decoder (see the same avoidance in train_whisper.prepare). soundfile
    handles the wav/flac case directly; anything else falls back to ffmpeg,
    which must be on PATH."""
    try:
        audio, sr = sf.read(io.BytesIO(raw["bytes"]), dtype="float32")
    except Exception:
        suffix = Path(raw.get("path") or "clip.audio").suffix or ".audio"
        src_fd, src_path = tempfile.mkstemp(suffix=suffix)
        dst_fd, dst_path = tempfile.mkstemp(suffix=".wav")
        try:
            with open(src_fd, "wb") as f:
                f.write(raw["bytes"])
            import os
            os.close(dst_fd)  # ffmpeg will open/write dst_path itself
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src_path,
                 "-ar", str(SAMPLE_RATE), "-ac", "1", dst_path],
                check=True,
            )
            audio, sr = sf.read(dst_path, dtype="float32")
        finally:
            import os
            os.unlink(src_path)
            os.unlink(dst_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def resample_if_needed(audio, sr):
    if sr == SAMPLE_RATE:
        return audio
    import os
    src_fd, src_path = tempfile.mkstemp(suffix=".wav")
    dst_fd, dst_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.close(src_fd)
        os.close(dst_fd)
        sf.write(src_path, audio, sr)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src_path,
             "-ar", str(SAMPLE_RATE), "-ac", "1", dst_path],
            check=True,
        )
        audio, _ = sf.read(dst_path, dtype="float32")
    finally:
        os.unlink(src_path)
        os.unlink(dst_path)
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
    parser.add_argument(
        "--id-column", default="auto",
        help="column identifying the source recording (auto | none | NAME). When present, "
        "validation holds out whole recordings instead of tail-slicing. 'auto' probes "
        "the first row for video_id/episode_id/id.",
    )
    parser.add_argument("--min-sec", type=float, default=0.3)
    parser.add_argument("--max-sec", type=float, default=28.0)
    parser.add_argument(
        "--drop-annotations", action="store_true",
        help="drop rows whose text contains '*'. Some sources (kouman) wrap "
        "non-speech stage directions in asterisks -- '* applause *', on-screen "
        "text -- whose words are NOT in the audio. They are ~100%% WER no matter "
        "how good the model gets, and training on them teaches hallucination. "
        "Asterisks do not otherwise occur in these transcripts, so this is a "
        "safe marker; brackets are NOT, since '[host:]' is noise but '(BPM)' "
        "is spoken",
    )
    parser.add_argument(
        "--max-chars-per-sec", type=float, default=None,
        help="drop rows whose text is too long to have been spoken in the clip's "
        "duration -- catches on-screen text and audio/text misalignment "
        "regardless of punctuation convention. The clean sources here sit at a "
        "p95 of 15-20; 25 is conservative",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = [], []
    for spec in args.repos:
        repo, text_col, split = parse_repo_spec(spec)
        tag = repo.split("/")[-1]
        print(f"[{tag}] streaming split={split} (text_col={text_col}) ...", file=sys.stderr)
        # HF_TOKEN is unset for the public MohammadGholizadeh repos this was
        # written against; token=None there is the same unauthenticated
        # request datasets would make anyway. Only your own private repo needs it.
        ds = load_dataset(repo, split=split, streaming=True,
                          token=os.environ.get("HF_TOKEN"))
        ds = ds.cast_column("audio", Audio(decode=False))

        has_named_val = split in ("validation", "val", "test")
        id_col = None if args.id_column == "none" else args.id_column
        rows, total_sec = [], 0.0
        n_scanned = n_empty = n_dur = n_annot = n_fast = 0
        hit_budget = False
        for i, row in enumerate(ds):
            if args.max_seconds_per_repo and total_sec >= args.max_seconds_per_repo:
                hit_budget = True
                break
            if id_col == "auto":  # probe the first row we actually see
                id_col = next((c for c in ("video_id", "episode_id", "id") if c in row), None)
                print(f"[{tag}] recording id column: {id_col or 'none -- will tail-slice'}",
                      file=sys.stderr)
            n_scanned += 1
            text = normalize_fa(row.get(text_col) or "")
            if not text:
                n_empty += 1
                continue
            # cheap text-only reject, before paying for the decode
            if args.drop_annotations and "*" in text:
                n_annot += 1
                continue
            audio, sr = decode_row_audio(row["audio"])
            audio = resample_if_needed(audio, sr)
            dur = len(audio) / SAMPLE_RATE
            if dur < args.min_sec or dur > args.max_sec:
                n_dur += 1
                continue
            if args.max_chars_per_sec and len(text) / dur > args.max_chars_per_sec:
                n_fast += 1
                continue
            name = f"{tag}_{i:06d}.wav"
            sf.write(clips_dir / name, audio, SAMPLE_RATE)
            rows.append({"audio": name, "text": text, "source": tag, "duration": round(dur, 3),
                         "_gid": str(row.get(id_col)) if id_col else None})
            total_sec += dur
            if len(rows) % 500 == 0:
                print(f"[{tag}] {len(rows)} clips, {total_sec / 3600:.2f}h", file=sys.stderr)

        if has_named_val:
            mode = "named val/test split"
        elif id_col and any(r["_gid"] for r in rows):
            mode = f"held out by {id_col}"
        else:
            mode = "train, tail-sliced for val"
        print(f"[{tag}] done: {len(rows)} clips, {total_sec / 3600:.2f}h ({mode})",
              file=sys.stderr)
        # Say WHY the loop ended and what it discarded. Without this a build that
        # stopped at its budget after 4% of the repo is indistinguishable from one
        # that read the whole thing and threw most of it away -- the two call for
        # opposite responses (raise the budget vs loosen the filters).
        print(f"[{tag}] scanned {n_scanned} rows -> kept {len(rows)}"
              f" (dropped {n_empty} empty-text, {n_dur} outside "
              f"{args.min_sec}-{args.max_sec}s, {n_annot} annotation, "
              f"{n_fast} over {args.max_chars_per_sec} chars/s)", file=sys.stderr)
        if hit_budget:
            print(f"[{tag}] STOPPED at the --max-seconds-per-repo budget "
                  f"({args.max_seconds_per_repo:.0f}s) -- the repo has more available; "
                  f"raise it to pull more", file=sys.stderr)
        else:
            print(f"[{tag}] reached the end of the split", file=sys.stderr)

        if has_named_val:
            val_rows.extend(rows)
        elif id_col and any(r["_gid"] for r in rows):
            # Hold out WHOLE recordings by video_id. A tail slice
            # would leak here: this repo's train split is shuffled across videos
            # (consecutive rows carry different ids), so its last N rows belong to
            # videos that are also in the training portion -- the model would be
            # scored on recordings it trained on, and the WER would read better
            # than the model is.
            gids = sorted({r["_gid"] for r in rows})
            n_val = max(1, round(len(gids) * args.val_fraction)) if len(gids) > 1 else 0
            val_gids = set(gids[:n_val])
            val_rows.extend(r for r in rows if r["_gid"] in val_gids)
            train_rows.extend(r for r in rows if r["_gid"] not in val_gids)
            print(f"[{tag}] {len(gids)} recordings -> {len(val_gids)} held out for validation",
                  file=sys.stderr)
        else:
            # No id column: the rows are consecutive utterances of a continuous
            # recording, so the tail IS later material rather than a random
            # sample of the same content. Same speaker on both sides, but that is
            # the best available without re-diarizing the source.
            cut = max(1, int(len(rows) * args.val_fraction)) if rows else 0
            val_rows.extend(rows[-cut:] if cut else [])
            train_rows.extend(rows[:-cut] if cut else rows)

    if not train_rows and not val_rows:
        sys.exit("no clips produced")

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows + val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # _gid is bookkeeping for the split above; drop it so the saved columns match
    # what blend_datasets.py emits
    strip = lambda rs: [{k: v for k, v in r.items() if k != "_gid"} for r in rs]
    # only emit splits that actually have rows -- pulling a repo's own val split
    # on its own (--repo ...:transcription:val) is a legitimate way to source a
    # clean held-out set, and that build has no train rows by construction
    splits = {}
    if train_rows:
        splits["train"] = Dataset.from_list(strip(train_rows))
    if val_rows:
        splits["validation"] = Dataset.from_list(strip(val_rows))
    DatasetDict(splits).save_to_disk(str(out_dir / "hf_dataset"))

    total_h = sum(r["duration"] for r in train_rows + val_rows) / 3600
    print(f"\ntotal: {len(train_rows) + len(val_rows)} clips, {total_h:.2f}h "
          f"(train={len(train_rows)}, val={len(val_rows)})")
    print(f"dataset -> {out_dir / 'hf_dataset'}")
    print(f"clips   -> {clips_dir}")


if __name__ == "__main__":
    main()
    # Streaming leaves background threads alive (the fsspec/aiohttp readers
    # behind load_dataset(streaming=True)), and on interpreter shutdown one of
    # them can touch the GIL after finalization has begun:
    #     Fatal Python error: PyGILState_Release: auto-releasing thread-state,
    #     but no thread-state for this thread
    # The process dies with SIGABRT *after* every clip, the manifest and the
    # dataset are written and this script has printed its summary -- so the
    # build has fully succeeded and the caller still sees a crash. Everything is
    # flushed by this point, so skip interpreter finalization rather than race
    # it. An exception inside main() still propagates normally and exits
    # non-zero; only a completed build takes this path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

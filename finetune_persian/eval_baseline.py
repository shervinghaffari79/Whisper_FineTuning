#!/usr/bin/env python3
"""
Measure WER/CER of a CTranslate2 Whisper model on a held-out split -- run it
on the CURRENT production model before training to get a baseline, then again
on the fine-tuned model to see whether fine-tuning actually helped. Without
the baseline you cannot tell improvement from regression.

Decoding uses the same parameters asr_engine.transcribe_chunk does, so the
number reflects what you would actually ship.

Usage:
    # local disk (hf_build_dataset.py output)
    python eval_baseline.py --model ./models/whisper-large-v3-persian-ct2-int8 \
        --dataset ./sm_data/collection/hf_dataset --limit 200

    # Hugging Face Hub (audio embedded in the dataset -- no clips dir needed)
    python eval_baseline.py --model ./models/whisper-large-v3-persian-ct2-int8 \
        --dataset shervingh2000/behpardaz --limit 200

Requires: faster-whisper, jiwer, datasets, soundfile.
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import jiwer
import soundfile as sf
from datasets import Audio, load_dataset, load_from_disk

# Persian orthography varies across sources; without normalizing both sides,
# WER measures typography rather than recognition quality. Shared with
# train_whisper.compute_metrics so the in-training number and this one are
# measuring the same thing -- see fa_text.py.
from fa_text import normalize_for_wer as normalize_fa
from model_paths import default_ct2_model_dir


def load_split(spec: str, split: str):
    """spec is a local disk path (a DatasetDict built by hf_build_dataset.py,
    "audio" column = bare filenames next to a clips/ dir) or a Hugging Face
    Hub repo id (audio embedded, decode=False to avoid the Audio feature's
    torchcodec-backed decoder -- same reasoning as train_whisper.load_dataset_dict).
    Returns (Dataset, is_hub)."""
    if Path(spec).exists():
        return load_from_disk(spec)[split], False
    ds = load_dataset(spec, split=split)
    return ds.cast_column("audio", Audio(decode=False)), True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=default_ct2_model_dir(),
        help="CTranslate2 model directory (default: CT2_MODEL_DIR, then the project-local model)",
    )
    parser.add_argument("--dataset", default="./sm_data/dataset/hf_dataset")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--clips-dir", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="score only N clips, drawn at random but stable for a given "
                        "--seed so baseline and fine-tuned runs see the same subset")
    parser.add_argument("--seed", type=int, default=42, help="selects which clips --limit draws")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--save-predictions", default=None, help="write per-clip results to JSONL")
    args = parser.parse_args()

    import ctranslate2
    from faster_whisper import WhisperModel

    device = args.device
    if device == "auto":
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = args.compute_type or ("int8_float16" if device == "cuda" else "int8")

    # faster-whisper falls back to treating --model as a Hugging Face repo id
    # whenever the local directory is absent, so a typo'd or wrong path fails
    # with "Repo id must be in the form 'repo_name' or 'namespace/repo_name'"
    # -- an error that says nothing about the actual problem. Catch it here
    # while we can still name the path we looked for.
    looks_local = (os.sep in args.model or "/" in args.model
                   or args.model.startswith(".") or Path(args.model).is_absolute())
    if looks_local and not Path(args.model).is_dir():
        sys.exit(f"model directory not found: {Path(args.model).resolve()}\n"
                 f"Pull it from the Modal volume first (see the README's Quick start):\n"
                 f"  modal volume get whisper-persian-data models/whisper-persian-ct2-int8 ./models\n"
                 f"Pass a bare name like 'small' to pull from Hugging Face instead.")

    ds, is_hub = load_split(args.dataset, args.split)
    clips_dir = None
    if not is_hub:
        clips_dir = Path(args.clips_dir) if args.clips_dir else Path(args.dataset).parent / "clips"
        if not clips_dir.is_dir():
            sys.exit(f"clips dir not found: {clips_dir.resolve()} (pass --clips-dir)")
    # Shuffle before capping. hf_build_dataset.py emits clips grouped by video and
    # ordered by position within it, so a plain select(range(n)) scores the opening n clips
    # of a single recording -- one speaker, one acoustic condition. The fixed
    # seed keeps the subset identical between the baseline run and the
    # fine-tuned run, which is the only thing that makes the two comparable.
    n_total = len(ds)
    if args.limit and args.limit < n_total:
        ds = ds.shuffle(seed=args.seed).select(range(args.limit))

    print(f"model: {args.model}  ({device}/{compute_type})", file=sys.stderr)
    print(f"split: {args.split}  n={len(ds)} of {n_total}"
          + (f"  (random subset, seed={args.seed})" if len(ds) < n_total else ""),
          file=sys.stderr)
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    refs, hyps, rows = [], [], []
    t0 = time.time()
    for i, ex in enumerate(ds):
        if is_hub:
            audio, _sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        else:
            audio, _sr = sf.read(clips_dir / ex["audio"], dtype="float32")
        segments, _ = model.transcribe(
            audio, language="fa", task="transcribe", beam_size=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], compression_ratio_threshold=2.4,
            no_speech_threshold=0.45, condition_on_previous_text=False,
            vad_filter=False, word_timestamps=False,
        )
        hyp = " ".join(s.text for s in segments)
        ref_n, hyp_n = normalize_fa(ex["text"]), normalize_fa(hyp)
        if not ref_n:
            continue
        refs.append(ref_n)
        hyps.append(hyp_n)
        # hf_build_dataset.py/blend_datasets.py name the origin "source"; some
        # builds have no such column and identify the recording by "video_id"
        # instead. Falling back keeps the per-origin breakdown working for
        # both -- exactly the split you want when the held-out set is a
        # handful of distinct speakers rather than one blended corpus.
        # ex["audio"] is {"bytes", "path"} for a Hub row (decode=False, see
        # load_split) -- not JSON-serializable and not a useful identifier, so
        # record its path (or a row number if even that's absent) instead
        audio_id = ex["audio"].get("path") or f"row{i}" if is_hub else ex["audio"]
        rows.append({"audio": audio_id, "ref": ex["text"], "hyp": hyp,
                     "source": ex.get("source") or ex.get("video_id")})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ds)}  ({(time.time() - t0) / (i + 1):.2f}s/clip)", file=sys.stderr)

    if not refs:
        sys.exit("no scorable clips")

    print()
    print(f"clips scored : {len(refs)}")
    print(f"WER          : {100 * jiwer.wer(refs, hyps):.2f}%")
    print(f"CER          : {100 * jiwer.cer(refs, hyps):.2f}%")
    print(f"wall time    : {time.time() - t0:.1f}s")

    # A blended validation set (blend_datasets.py) mixes populations, and the
    # aggregate over a set that grew is NOT comparable to a number measured
    # before the growth -- adding easier or harder voices moves it on its own.
    # The per-source rows are what stay comparable run to run, so print them
    # whenever the dataset carries a source column.
    sources = sorted({r["source"] for r in rows if r.get("source")})
    if len(sources) > 1:
        print("\n--- by source ---")
        print(f"{'source':24s} {'clips':>6s} {'WER':>8s} {'CER':>8s}")
        for src in sources:
            idx = [i for i, r in enumerate(rows) if r.get("source") == src]
            s_refs = [refs[i] for i in idx]
            s_hyps = [hyps[i] for i in idx]
            print(f"{src:24s} {len(idx):6d} {100 * jiwer.wer(s_refs, s_hyps):7.2f}% "
                  f"{100 * jiwer.cer(s_refs, s_hyps):7.2f}%")
    print("\n--- sample predictions ---")
    for row in rows[:5]:
        print(f"  ref: {row['ref']}")
        print(f"  hyp: {row['hyp']}\n")

    if args.save_predictions:
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"predictions -> {args.save_predictions}")


if __name__ == "__main__":
    main()

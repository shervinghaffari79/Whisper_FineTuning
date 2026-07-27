#!/usr/bin/env python3
"""
Measure WER/CER of a CTranslate2 Whisper model on a held-out split -- run it
on the CURRENT production model before training to get a baseline, then again
on the fine-tuned model to see whether fine-tuning actually helped. Without
the baseline you cannot tell improvement from regression.

Decoding uses the same parameters asr_engine.transcribe_chunk does, so the
number reflects what you would actually ship.

Usage:
    python eval_baseline.py --model ./models/whisper-large-v3-persian-ct2-int8 \
        --dataset ./sm_data/dataset/hf_dataset --limit 200

Requires: faster-whisper, jiwer, datasets, soundfile.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import jiwer
import soundfile as sf
from datasets import load_from_disk

# Persian orthography varies across sources; without normalizing both sides,
# WER measures typography rather than recognition quality. Shared with
# train_whisper.compute_metrics so the in-training number and this one are
# measuring the same thing -- see fa_text.py.
from fa_text import normalize_for_wer as normalize_fa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="CTranslate2 model directory")
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
                 f"CT2 models live next to asr_engine.py, not next to this script -- "
                 f"from finetune_persian/ that is ../whiper/Whisper/models/<name>.\n"
                 f"Pass a bare name like 'small' to pull from Hugging Face instead.")

    clips_dir = Path(args.clips_dir) if args.clips_dir else Path(args.dataset).parent / "clips"
    if not clips_dir.is_dir():
        sys.exit(f"clips dir not found: {clips_dir.resolve()} (pass --clips-dir)")
    ds = load_from_disk(args.dataset)[args.split]
    # Shuffle before capping. sm_04 emits clips grouped by video and ordered by
    # position within it, so a plain select(range(n)) scores the opening n clips
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
        rows.append({"audio": ex["audio"], "ref": ex["text"], "hyp": hyp})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ds)}  ({(time.time() - t0) / (i + 1):.2f}s/clip)", file=sys.stderr)

    if not refs:
        sys.exit("no scorable clips")

    print()
    print(f"clips scored : {len(refs)}")
    print(f"WER          : {100 * jiwer.wer(refs, hyps):.2f}%")
    print(f"CER          : {100 * jiwer.cer(refs, hyps):.2f}%")
    print(f"wall time    : {time.time() - t0:.1f}s")
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

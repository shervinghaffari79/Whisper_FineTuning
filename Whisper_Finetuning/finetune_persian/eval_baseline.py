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
import re
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import soundfile as sf
from datasets import load_from_disk

# Persian orthography varies across sources; without normalizing both sides,
# WER measures typography rather than recognition quality.
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "ا"})
_DIACRITICS = re.compile(r"[ً-ْـ]")          # harakat + tatweel
_ZWNJ = re.compile(r"[‌‎‏]")  # ZWNJ and bidi marks
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_fa(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _DIACRITICS.sub("", text)
    text = _ZWNJ.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    for i, (fa, ar) in enumerate(zip("۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩")):
        text = text.replace(fa, str(i)).replace(ar, str(i))
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="CTranslate2 model directory")
    parser.add_argument("--dataset", default="./sm_data/dataset/hf_dataset")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--clips-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
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

    clips_dir = Path(args.clips_dir) if args.clips_dir else Path(args.dataset).parent / "clips"
    ds = load_from_disk(args.dataset)[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"model: {args.model}  ({device}/{compute_type})", file=sys.stderr)
    print(f"split: {args.split}  n={len(ds)}", file=sys.stderr)
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

#!/usr/bin/env python3
"""Diagnose a CTranslate2 Whisper model that transcribes into the wrong
language.

Symptom this exists for: eval_baseline.py reports WER > 100% and the
hypotheses come back as English translations of Persian audio, even though
transcribe() is called with language="fa", task="transcribe".

That combination almost always means the decoder is not receiving the control
tokens we think it is -- usually because the CT2 directory's tokenizer assets
do not match the weights they sit next to, so <|fa|> and <|transcribe|> map to
different ids than the model was trained with. This script prints the facts
needed to tell that apart from "the model is simply bad at Persian":

  1. what is actually in the model directory
  2. whether faster-whisper considers the model multilingual
  3. what language the model detects on its own
  4. whether task="transcribe" and task="translate" differ at all
     (identical output = the task token is being ignored)

Usage:
    python diagnose_ct2.py --model ./models/whisper-large-v3-persian-ct2-int8
"""
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_from_disk

from model_paths import default_ct2_model_dir


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
    parser.add_argument("--n", type=int, default=3, help="clips to probe")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_dir = Path(args.model)
    print("=" * 72)
    print(f"1. MODEL DIRECTORY  {model_dir.resolve()}")
    print("=" * 72)
    if not model_dir.is_dir():
        sys.exit(f"not a directory: {model_dir.resolve()}")
    for p in sorted(model_dir.iterdir()):
        size = p.stat().st_size
        print(f"   {p.name:<34} {size / 1e6:>10.2f} MB")

    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        print("\n   config.json:")
        for k, v in cfg.items():
            # suppress_ids is hundreds of entries and tells us nothing here
            shown = f"<{len(v)} entries>" if isinstance(v, list) and len(v) > 12 else v
            print(f"     {k}: {shown}")
        # the tell: an English-only vocabulary cannot express <|fa|>, and
        # faster-whisper silently forces English when it sees one
        print("\n   -> multilingual:", cfg.get("multilingual", "NOT SET"))
    else:
        print("\n   WARNING: no config.json -- conversion likely incomplete")

    for name in ("tokenizer.json", "vocabulary.json", "vocabulary.txt"):
        p = model_dir / name
        print(f"   {name:<20} {'present' if p.exists() else 'MISSING'}")

    import ctranslate2
    from faster_whisper import WhisperModel

    device = args.device
    if device == "auto":
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"

    print("\n" + "=" * 72)
    print(f"2. LOADED MODEL  ({device}/{compute_type})")
    print("=" * 72)
    model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
    print(f"   is_multilingual : {getattr(model, 'is_multilingual', 'n/a')}")
    print(f"   num_languages   : {getattr(model, 'num_languages', 'n/a')}")
    # if this is False, faster-whisper ignores language="fa" entirely and the
    # English output is fully explained
    if getattr(model, "is_multilingual", True) is False:
        print("   *** model reports English-only: language='fa' is being IGNORED ***")

    clips_dir = Path(args.clips_dir) if args.clips_dir else Path(args.dataset).parent / "clips"
    ds = load_from_disk(args.dataset)[args.split]

    print("\n" + "=" * 72)
    print("3. PER-CLIP PROBE")
    print("=" * 72)
    common = dict(beam_size=5, temperature=[0.0], condition_on_previous_text=False,
                  vad_filter=False, word_timestamps=False)
    for i in range(min(args.n, len(ds))):
        ex = ds[i]
        audio, _sr = sf.read(clips_dir / ex["audio"], dtype="float32")
        print(f"\n-- clip {i}: {ex['audio']}  ({len(audio) / 16000:.1f}s)")
        print(f"   ref                     : {ex['text'][:110]}")

        # what does the model think the language is, unprompted?
        segs, info = model.transcribe(audio, **common)
        text = " ".join(s.text for s in segs).strip()
        print(f"   auto-detect             : lang={info.language} "
              f"p={info.language_probability:.3f}")
        print(f"     -> {text[:110]}")

        segs, _ = model.transcribe(audio, language="fa", task="transcribe", **common)
        as_transcribe = " ".join(s.text for s in segs).strip()
        print(f"   language=fa transcribe  : {as_transcribe[:110]}")

        segs, _ = model.transcribe(audio, language="fa", task="translate", **common)
        as_translate = " ".join(s.text for s in segs).strip()
        print(f"   language=fa translate   : {as_translate[:110]}")

        if as_transcribe == as_translate:
            print("   *** transcribe and translate are IDENTICAL -- the task token "
                  "is not reaching the decoder ***")

    print("\n" + "=" * 72)
    print("READING THE RESULT")
    print("=" * 72)
    print("""   is_multilingual False        -> wrong/English-only conversion; reconvert
   transcribe == translate      -> tokenizer assets do not match the weights
   auto-detect says 'en' on fa  -> same as above, or the weights are not the
                                   Persian checkpoint they are named after
   auto-detect says 'fa' and
     transcribe returns Persian -> the model is fine; the fault is in how
                                   eval_baseline called it""")


if __name__ == "__main__":
    main()

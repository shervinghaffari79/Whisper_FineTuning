#!/usr/bin/env python3
"""
Quick look at what a mid-training checkpoint is actually generating, without
waiting for the next scheduled eval. Useful when eval_wer looks wrong (e.g.
over 100%) and you want to know whether that's early-training noise or a real
decoding bug (repetition looping, wrong language token, etc.) before deciding
to let the run continue or kill it.

Defaults to CPU: if train_whisper.py is still running in another window, it
already holds most of the T4's 16GB, and loading a second full copy of
large-v3 onto the same GPU risks OOMing the training run. Pass --device cuda
only once training has stopped or you know there's headroom.

Usage:
    python peek_checkpoint.py --checkpoint ./run1/checkpoint-200 --n 5
"""
import argparse

import soundfile as sf
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="a run1/checkpoint-N dir")
    parser.add_argument("--base-model", default="AmirMohseni/whisper-large-v3-persian-bf16")
    parser.add_argument("--dataset", default="./sm_data/dataset/hf_dataset")
    parser.add_argument("--clips-dir", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--device", default="cpu", help="cpu (safe alongside a running "
                        "training job) or cuda (faster, but competes for T4 VRAM)")
    args = parser.parse_args()

    from pathlib import Path
    clips_dir = Path(args.clips_dir) if args.clips_dir else Path(args.dataset).parent / "clips"

    processor = WhisperProcessor.from_pretrained(args.base_model, language="persian", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model).float().to(args.device)
    model = PeftModel.from_pretrained(base, args.checkpoint).to(args.device).eval()

    ds = load_from_disk(args.dataset)[args.split].select(range(args.n))
    for ex in ds:
        audio, sr = sf.read(clips_dir / ex["audio"], dtype="float32")
        feats = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features
        feats = feats.to(args.device)
        with torch.no_grad():
            ids = model.generate(feats, language="persian", task="transcribe", max_new_tokens=225)
        hyp = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        print(f"clip : {ex['audio']}")
        print(f"ref  : {ex['text']}")
        print(f"hyp  : {hyp}")
        print(f"  (hyp tokens: {ids.shape[-1]}, ref chars: {len(ex['text'])})")
        print()


if __name__ == "__main__":
    main()

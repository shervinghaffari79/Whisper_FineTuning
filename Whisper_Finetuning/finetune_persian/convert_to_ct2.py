#!/usr/bin/env python3
"""
Step 6 -- convert a fine-tuned HF Whisper checkpoint into the CTranslate2 int8
format that asr_engine.py serves in production.

A LoRA checkpoint (train_whisper.py --use-lora) is detected automatically and
merged into the base weights first: CTranslate2 has no notion of adapters, so
an unmerged adapter directory cannot be converted or served.

Usage:
    python convert_to_ct2.py ./run1/final ./models/whisper-large-v3-persian-ct2-int8-v2

Then point the backend at it -- no code changes needed, asr_engine.py reads
this env var:
    $env:CT2_MODEL_DIR="C:\\path\\to\\...-v2"     (PowerShell)
    export CT2_MODEL_DIR=/path/to/...-v2          (bash)

Requires: ctranslate2 (provides the ct2-transformers-converter CLI), plus
transformers/peft for the LoRA merge path.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# mirror the asset set of the existing production CT2 model so the output is a
# drop-in replacement; files absent from the checkpoint are skipped, not fatal
ASSET_FILES = [
    "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
    "generation_config.json", "special_tokens_map.json", "vocab.json",
    "merges.txt", "normalizer.json", "added_tokens.json",
]


def merge_lora(adapter_dir: Path, merged_dir: Path) -> Path:
    """Fold LoRA adapter weights into the base model, producing a plain HF
    checkpoint that ct2-transformers-converter can consume."""
    from peft import PeftConfig, PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_name = peft_config.base_model_name_or_path
    print(f"merging LoRA adapter into base model {base_name}...", file=sys.stderr)

    base = WhisperForConditionalGeneration.from_pretrained(base_name)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))

    # the adapter dir carries the processor assets; fall back to the base repo
    # for any it lacks, so the merged checkpoint is self-contained
    for name in ASSET_FILES:
        if (adapter_dir / name).exists() and not (merged_dir / name).exists():
            shutil.copy2(adapter_dir / name, merged_dir / name)
    if not any((merged_dir / n).exists() for n in ("tokenizer.json", "vocab.json")):
        WhisperProcessor.from_pretrained(base_name).save_pretrained(str(merged_dir))
    print(f"merged checkpoint -> {merged_dir}", file=sys.stderr)
    return merged_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_model_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--quantization", default="int8_float16",
                        help="int8_float16 for CUDA (T4), int8 for CPU")
    parser.add_argument("--merged-dir", default=None,
                        help="where to write the merged checkpoint when converting a "
                        "LoRA adapter (default: <out_dir>_merged)")
    args = parser.parse_args()

    src = Path(args.hf_model_dir)
    if (src / "adapter_config.json").exists():
        merged = Path(args.merged_dir) if args.merged_dir else Path(str(args.out_dir) + "_merged")
        src = merge_lora(src, merged)

    copy_files = [name for name in ASSET_FILES if (src / name).exists()]
    cmd = [
        "ct2-transformers-converter",
        "--model", str(src),
        "--output_dir", args.out_dir,
        "--quantization", args.quantization,
        "--copy_files", *copy_files,
    ]
    print("running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    print(f"converted -> {args.out_dir}")


if __name__ == "__main__":
    main()

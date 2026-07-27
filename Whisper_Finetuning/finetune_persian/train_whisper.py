#!/usr/bin/env python3
"""
Step 5 -- fine-tune the Persian Whisper checkpoint on the dataset built by
sm_04_build_dataset.py. Meant for the Windows/T4 box.

Continues from AmirMohseni/whisper-large-v3-persian-bf16 (the same base the
production CT2 model was converted from) so the result stays a drop-in
upgrade of what asr_engine.py already serves.

VRAM note: full fine-tuning whisper-large-v3 needs ~22GB of weights +
optimizer state and will OOM a 16GB T4. Use --use-lora (recommended: ~6-8GB,
and it regularizes against overfitting on small datasets), or at minimum
--optim adamw_bnb_8bit.

Usage:
    python train_whisper.py --dataset ./sm_data/dataset/hf_dataset --out ./run1 --use-lora

Requires: torch (CUDA build), transformers, datasets, accelerate, evaluate,
jiwer, soundfile, and peft when --use-lora is set.
"""
import argparse
import sys
from pathlib import Path

import datasets as hf_datasets
import evaluate
import numpy as np
import soundfile as sf
import torch
import transformers
from datasets import load_from_disk
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from fa_text import normalize_for_wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./sm_data/dataset/hf_dataset")
    parser.add_argument("--base-model", default="AmirMohseni/whisper-large-v3-persian-bf16")
    parser.add_argument("--out", default="./run1")
    parser.add_argument("--language", default="persian")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="default 1e-5 for full fine-tuning, 1e-3 with --use-lora (adapters need a "
        "much larger LR than full FT -- 1e-5 barely moves them)",
    )
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument(
        "--optim", default="adamw_torch",
        help="use adamw_bnb_8bit (needs bitsandbytes) to fit large-v3 on a 16GB T4 -- "
        "plain adamw needs ~22GB of optimizer state alone",
    )
    parser.add_argument(
        "--use-lora", action="store_true",
        help="parameter-efficient fine-tuning: trains ~1%% of params (fits a 16GB T4 "
        "comfortably) and regularizes against overfitting on small datasets. Produces "
        "an adapter that convert_to_ct2.py merges into the base weights before conversion.",
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--progress", choices=["auto", "always", "never"], default="auto",
        help="auto (default) shows a live progress bar on a terminal and falls back to "
        "periodic step logs when output is piped to a file -- tqdm writes one line per "
        "update off-TTY, which floods a log; 'never' forces the compact log form",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="for smoke-testing")
    parser.add_argument(
        "--max-eval-samples", type=int, default=None,
        help="cap the validation set (a randomly drawn but seed-stable subset). Eval "
        "runs generate() over every clip and easily costs more wall clock than the "
        "training it is measuring -- 128-256 clips track the full-set WER closely "
        "enough to pick a checkpoint",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="stop after N consecutive evals with no WER improvement (0 disables). "
        "Cheap insurance on a small dataset, where WER typically bottoms out long "
        "before the epoch budget runs out and the remaining hours only overfit",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds training and the --max-*-samples subsets")
    parser.add_argument(
        "--clips-dir", default=None,
        help="dir holding the audio clips referenced by the dataset; defaults to a "
        "'clips' folder next to --dataset (where sm_04 puts them)",
    )
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 1e-3 if args.use_lora else 1e-5

    # Progress reporting: tqdm redraws in place on a terminal, but emits a NEW
    # LINE per update when stdout is a pipe/log file -- which floods the output
    # with thousands of lines over a real run. So keep the live bar only when
    # attached to a TTY, and fall back to periodic one-line step logs otherwise.
    interactive = sys.stderr.isatty()
    show_bar = {"auto": interactive, "always": True, "never": False}[args.progress]
    # silence the per-batch INFO chatter that would otherwise interleave with the bar
    transformers.utils.logging.set_verbosity_warning()
    hf_datasets.utils.logging.set_verbosity_warning()
    if not show_bar:
        hf_datasets.disable_progress_bars()

    clips_dir = Path(args.clips_dir) if args.clips_dir else Path(args.dataset).parent / "clips"
    if not clips_dir.exists():
        raise SystemExit(f"clips dir not found: {clips_dir} (pass --clips-dir)")

    ds = load_from_disk(args.dataset)
    # Subsample by SHUFFLING first, with a fixed seed. sm_04 emits clips grouped
    # by video and ordered by position within it, so a plain .select(range(n))
    # would hand back the opening n clips of whichever video sorts first -- one
    # speaker, one topic, one recording condition. Capping eval that way makes
    # the WER faster and meaningless at the same time. The fixed seed keeps the
    # subset identical across runs so the numbers stay comparable.
    if args.max_train_samples and args.max_train_samples < len(ds["train"]):
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    if args.max_eval_samples and args.max_eval_samples < len(ds["validation"]):
        ds["validation"] = (ds["validation"].shuffle(seed=args.seed)
                            .select(range(args.max_eval_samples)))
    print(f"train={len(ds['train'])} clips, validation={len(ds['validation'])} clips",
          file=sys.stderr)

    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task="transcribe")

    def prepare(batch):
        # read the clip with soundfile rather than datasets' Audio feature:
        # the Audio feature now requires torchcodec, whose bundled FFmpeg libs
        # don't link on every platform, and we already know these are 16kHz mono
        audio, sr = sf.read(clips_dir / batch["audio"], dtype="float32")
        batch["input_features"] = processor.feature_extractor(
            audio, sampling_rate=sr
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds["train"].column_names, num_proc=1,
                desc="extracting log-mel features")

    class DataCollatorSpeechSeq2SeqWithPadding:
        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding()
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        # predict_with_generate can hand back a tuple (sequences, scores, ...)
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        # np.where, not in-place: pred.label_ids is Trainer's own buffer and
        # mutating it corrupts anything that reads the batch after us
        label_ids = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id,
                             pred.label_ids)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Score on normalized text, the same normalization eval_baseline.py
        # applies, so the two WERs are comparable. Unnormalized, every comma
        # the model places differently is a full word substitution -- sm_04
        # attaches punctuation to the preceding word -- which buries the
        # recognition signal this metric is supposed to expose and, because
        # metric_for_best_model="wer", drags checkpoint selection with it.
        refs = [normalize_for_wer(s) for s in label_str]
        hyps = [normalize_for_wer(s) for s in pred_str]
        keep = [i for i, r in enumerate(refs) if r.strip()]
        if not keep:
            return {"wer": 100.0, "wer_raw": 100.0}
        refs_k = [refs[i] for i in keep]
        hyps_k = [hyps[i] for i in keep]
        return {
            "wer": 100 * wer_metric.compute(predictions=hyps_k, references=refs_k),
            # kept visible so the gap between the two shows how much of the
            # error is punctuation/orthography rather than recognition
            "wer_raw": 100 * wer_metric.compute(
                predictions=[pred_str[i] for i in keep],
                references=[label_str[i] for i in keep]),
        }

    # The checkpoint ships bf16 weights and recent transformers preserves the
    # checkpoint dtype on load. Two problems with that here: a T4 (sm_75) has no
    # native bf16, and Trainer's fp16 autocast does not wrap the generate() call
    # in prediction_step -- so eval hits fp32 inputs against bf16 weights and
    # dies in the first conv. Hold master weights in fp32 and let fp16=True do
    # the mixed-precision work through autocast.
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model).float()
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    # eval's predict_with_generate uses plain greedy decoding (no beam search,
    # no repetition guard) -- on a checkpoint that's still early in training,
    # that degenerates into looping the same token for dozens of steps on a
    # subset of clips. Those clips alone can drag eval_wer over 100% even when
    # the rest of the set is decoding fine, which is noise metric_for_best_model
    # and early stopping both read directly. Blocking repeated 3-grams costs
    # nothing on well-formed output and forecloses the loop.
    model.generation_config.no_repeat_ngram_size = 3
    model.config.use_cache = False  # required with gradient checkpointing

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        # gradient checkpointing freezes the frozen base's input grads, which
        # would stop gradients ever reaching the adapters -- re-enable them
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],  # standard Whisper LoRA target set
            bias="none",
        ))
        model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        optim=args.optim,
        gradient_checkpointing=True,
        # the default reentrant checkpointing breaks Whisper's backward pass
        # under torch>=2.4 ("backward through the graph a second time")
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        # with a live bar the per-step numbers are already on screen, so log
        # sparsely; without one these lines ARE the progress signal
        logging_steps=50 if show_bar else 10,
        disable_tqdm=not show_bar,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        seed=args.seed,
        # a PeftModel's forward signature hides "labels" from Trainer's
        # auto-detection, so name it explicitly or eval loss/WER come out empty
        label_names=["labels"] if args.use_lora else None,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.out + "/final")
    processor.save_pretrained(args.out + "/final")
    print(f"saved fine-tuned model -> {args.out}/final")


if __name__ == "__main__":
    main()

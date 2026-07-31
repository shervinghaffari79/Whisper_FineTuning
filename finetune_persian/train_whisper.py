#!/usr/bin/env python3
"""
Step 5 -- fine-tune the Persian Whisper checkpoint on a dataset built by
hf_build_dataset.py (local disk) or pushed straight to the Hugging Face Hub
(e.g. push_to_hub.py's output). Meant for the Windows/T4 box.

Continues from AmirMohseni/whisper-large-v3-persian-bf16 (the same base the
production CT2 model was converted from) so the result stays a drop-in
upgrade of what asr_engine.py already serves.

VRAM note: full fine-tuning whisper-large-v3 needs ~22GB of weights +
optimizer state and will OOM a 16GB T4. Use --use-lora (recommended: ~6-8GB,
and it regularizes against overfitting on small datasets), or at minimum
--optim adamw_bnb_8bit.

Usage:
    # local disk (a DatasetDict saved by hf_build_dataset.py, with a sibling
    # clips/ folder of audio files -- --dataset's "audio" column is filenames)
    python train_whisper.py --dataset ./sm_data/collection/hf_dataset --out ./run1 --use-lora

    # Hugging Face Hub (audio embedded in the dataset -- no clips dir needed)
    python train_whisper.py --dataset shervingh2000/behpardaz --out ./run1 --use-lora

Requires: torch (CUDA build), transformers, datasets, accelerate, evaluate,
jiwer, soundfile, and peft when --use-lora is set.
"""
import argparse
import io
import sys
from pathlib import Path

import datasets as hf_datasets
import evaluate
import numpy as np
import soundfile as sf
import torch
import transformers
from datasets import Audio, Features, Value, concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from fa_text import normalize_for_wer


def load_dataset_dict(spec: str):
    """spec is a local disk path (a DatasetDict saved by hf_build_dataset.py,
    "audio" column = bare filenames next to a clips/ dir) or a Hugging Face
    Hub repo id (e.g. shervingh2000/behpardaz -- "audio" column is an Audio
    feature with the clip bytes embedded, no clips dir applies).

    A repo id never resolves to an existing local path, so that's the switch:
    local disk if the path exists, Hub otherwise. Returns (DatasetDict, is_hub).

    Hub splits get decode=False cast onto "audio": the datasets library's own
    Audio decoder now requires torchcodec, whose bundled FFmpeg libs don't
    link on every platform (the same reason the local-disk path below reads
    clips with soundfile directly instead of the Audio feature). decode=False
    hands back {"bytes", "path"} instead, which make_prepare/make_batch_prepare
    decode with soundfile themselves, same as hf_build_dataset.decode_row_audio.
    """
    if Path(spec).exists():
        return load_from_disk(spec), False
    ds = load_dataset(spec)
    for split in ds:
        ds[split] = ds[split].cast_column("audio", Audio(decode=False))
    return ds, True


def resolve_train_to_paths(ds_train, is_hub: bool, clips_dir, cache_dir):
    """Normalize a train split's "audio" column to absolute file path strings,
    so heterogeneous sources (local-disk filenames, Hub-embedded bytes) can be
    concatenated into one Dataset and read by the ordinary local-disk code
    path in make_prepare/make_batch_prepare below with no changes -- pathlib
    joins an absolute right-hand side by replacing the left
    (Path("x") / "/abs/y" == Path("/abs/y")), so passing these rows through
    clips_dir / batch["audio"] resolves to the absolute path unchanged
    regardless of what clips_dir is.

    Hub rows get their embedded bytes written to cache_dir ONCE (skipped on
    repeat runs if the file already exists there -- put cache_dir on a
    persistent volume to avoid re-decoding every run).
    """
    # map() otherwise keeps the ORIGINAL declared Arrow feature type for a
    # column even when the function returns a different kind of value for it
    # -- a Hub source's "audio" (Audio(decode=False), i.e. a {bytes, path}
    # struct) stays typed that way even after being overwritten with plain
    # path strings, which fails concatenate_datasets against a local-disk
    # source's Value("string") "audio" column (and cast_column can't fix it
    # after the fact either -- struct->string isn't a valid Arrow cast).
    # Declaring the target schema up front avoids the mismatch entirely.
    out_features = Features({"audio": Value("string"), "text": Value("string")})
    to_drop = [c for c in ds_train.column_names if c not in ("audio", "text")]
    if not is_hub:
        return ds_train.map(lambda r: {"audio": str(clips_dir / r["audio"])},
                            remove_columns=to_drop, features=out_features)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _write(r, idx):
        path = cache_dir / f"{idx:07d}.wav"
        if not path.exists():
            audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            sf.write(path, audio, sr)
        return {"audio": str(path)}

    return ds_train.map(_write, with_indices=True, remove_columns=to_drop, features=out_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./sm_data/collection/hf_dataset",
                        help="local disk path (hf_build_dataset.py output) or a "
                        "Hugging Face Hub dataset repo id (e.g. shervingh2000/behpardaz)")
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
        "--init-adapter", default=None,
        help="warm-start from an existing LoRA adapter (e.g. run1/final) instead of a "
        "fresh random one -- for continuing training on new data without discarding "
        "what a prior run learned. Weights only: optimizer/scheduler state is NOT "
        "restored, so this seeds a new training run rather than resuming the old one. "
        "Requires --use-lora.",
    )
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
        help="dir holding the audio clips referenced by a LOCAL --dataset; defaults to "
        "a 'clips' folder next to it. Ignored for a Hub --dataset (audio is embedded).",
    )
    parser.add_argument(
        "--eval-dataset", default=None,
        help="use this dataset's 'validation' split for eval instead of --dataset's own "
        "(local path or Hub repo id, same rules as --dataset). For training on a broad "
        "corpus while still tracking WER on your real target-domain validation set -- "
        "otherwise the metric driving checkpoint selection and early stopping stops "
        "reflecting what you actually care about, and you can't tell a broad-domain "
        "gain from a target-domain regression.",
    )
    parser.add_argument(
        "--eval-clips-dir", default=None,
        help="clips dir for a LOCAL --eval-dataset; defaults to a 'clips' folder next "
        "to it. Ignored for a Hub --eval-dataset.",
    )
    parser.add_argument(
        "--extra-dataset", default=None,
        help="an ADDITIONAL dataset (local path or Hub repo id) whose train split gets "
        "concatenated onto --dataset's train split -- e.g. train on your own recordings "
        "plus a broad public corpus at once. Validation stays untouched (--dataset's own "
        "or --eval-dataset's), so checkpoint selection and W&B still track the one "
        "validation set you actually care about, not a blend.",
    )
    parser.add_argument(
        "--extra-clips-dir", default=None,
        help="clips dir for a LOCAL --extra-dataset; defaults to a 'clips' folder next "
        "to it. Ignored for a Hub --extra-dataset.",
    )
    parser.add_argument(
        "--extra-cache-dir", default=None,
        help="where to materialize a Hub --extra-dataset's embedded audio to local wav "
        "files (needed to concatenate with a local-disk --dataset); defaults to "
        "'hub_cache/<repo-id>' next to --out. Put this on a persistent volume -- rows "
        "already written there are skipped on the next run.",
    )
    parser.add_argument(
        "--features", choices=["auto", "lazy", "precompute"], default="auto",
        help="how log-mel features are produced. precompute caches every clip's features "
        "to disk up front (fast per step, but a large-v3 feature is 1.46 MiB, so 100k "
        "clips is ~146 GiB of cache). lazy computes them per batch and caches nothing -- "
        "measured overhead is ~0.03s/clip against a ~16s training step. auto picks lazy "
        "above --lazy-threshold clips.",
    )
    parser.add_argument("--lazy-threshold", type=int, default=20000,
                        help="clip count above which --features auto chooses lazy")
    parser.add_argument(
        "--dataloader-workers", type=int, default=0,
        help="parallel feature extraction for --features lazy. 0 keeps it in the training "
        "process (safe everywhere); 2-4 helps if extraction ever becomes the bottleneck, "
        "but worker spawn is slow on Windows",
    )
    parser.add_argument(
        "--report-to", nargs="+", default=["tensorboard"],
        help="Trainer logging backends, e.g. --report-to tensorboard wandb. wandb needs "
        "WANDB_API_KEY in the environment (and WANDB_PROJECT/--run-name to organize runs).",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="run name for backends that group by run (e.g. wandb); defaults to --out's basename",
    )
    args = parser.parse_args()

    if args.init_adapter and not args.use_lora:
        raise SystemExit("--init-adapter requires --use-lora")

    if args.lr is None:
        if args.init_adapter:
            # an adapter that already converged needs a gentler nudge than one
            # starting from a random init -- 1e-3 (the from-scratch default) risks
            # kicking it out of its current optimum before the new data's steps
            # are enough to bring it back down
            args.lr = 2e-4
        else:
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

    ds, dataset_is_hub = load_dataset_dict(args.dataset)
    clips_dir = None
    if not dataset_is_hub:
        clips_dir = Path(args.clips_dir) if args.clips_dir else Path(args.dataset).parent / "clips"
        if not clips_dir.exists():
            raise SystemExit(f"clips dir not found: {clips_dir} (pass --clips-dir)")

    eval_clips_dir = clips_dir
    if args.eval_dataset:
        eval_ds, eval_is_hub = load_dataset_dict(args.eval_dataset)
        eval_clips_dir = None
        if not eval_is_hub:
            eval_clips_dir = (Path(args.eval_clips_dir) if args.eval_clips_dir
                              else Path(args.eval_dataset).parent / "clips")
            if not eval_clips_dir.exists():
                raise SystemExit(f"eval clips dir not found: {eval_clips_dir} (pass --eval-clips-dir)")
        ds["validation"] = eval_ds["validation"]

    if args.extra_dataset:
        extra_ds, extra_is_hub = load_dataset_dict(args.extra_dataset)
        extra_clips_dir = None
        if not extra_is_hub:
            extra_clips_dir = (Path(args.extra_clips_dir) if args.extra_clips_dir
                              else Path(args.extra_dataset).parent / "clips")
            if not extra_clips_dir.exists():
                raise SystemExit(f"extra clips dir not found: {extra_clips_dir} (pass --extra-clips-dir)")
        extra_cache_dir = (Path(args.extra_cache_dir) if args.extra_cache_dir
                           else Path(args.out).parent / "hub_cache" / args.extra_dataset.replace("/", "_"))
        # primary also needs resolving to the same {audio: abs path, text} schema before
        # concatenation -- its own cache dir only matters if IT is a Hub dataset too
        primary_cache_dir = Path(args.out).parent / "hub_cache" / (
            args.dataset.replace("/", "_") if dataset_is_hub else "primary")
        n_primary = len(ds["train"])
        primary_train = resolve_train_to_paths(ds["train"], dataset_is_hub, clips_dir, primary_cache_dir)
        extra_train = resolve_train_to_paths(extra_ds["train"], extra_is_hub, extra_clips_dir, extra_cache_dir)
        ds["train"] = concatenate_datasets([primary_train, extra_train])
        # clips_dir / "<already-absolute-path>" resolves to that absolute path unchanged
        # (see resolve_train_to_paths docstring), so any placeholder works here
        clips_dir = Path(".")
        print(f"combined train: {n_primary} (--dataset) + {len(extra_train)} (--extra-dataset) "
              f"= {len(ds['train'])} clips", file=sys.stderr)

    # Subsample by SHUFFLING first, with a fixed seed. hf_build_dataset.py emits
    # clips grouped by video and ordered by position within it, so a plain .select(range(n))
    # would hand back the opening n clips of whichever video sorts first -- one
    # speaker, one topic, one recording condition. Capping eval that way makes
    # the WER faster and meaningless at the same time. The fixed seed keeps the
    # subset identical across runs so the numbers stay comparable.
    if args.max_train_samples and args.max_train_samples < len(ds["train"]):
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    if args.max_eval_samples and args.max_eval_samples < len(ds["validation"]):
        ds["validation"] = (ds["validation"].shuffle(seed=args.seed)
                            .select(range(args.max_eval_samples)))
    print(f"train={len(ds['train'])} clips, validation={len(ds['validation'])} clips"
          + (f"  (eval from {args.eval_dataset})" if args.eval_dataset else ""),
          file=sys.stderr)

    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task="transcribe")

    def make_prepare(audio_dir):
        def prepare(batch):
            if audio_dir is None:
                # Hub dataset: "audio" is {"bytes", "path"} (decode=False, see
                # load_dataset_dict) -- decode with soundfile ourselves, same
                # avoidance of the torchcodec-backed Audio decoder as below
                audio, sr = sf.read(io.BytesIO(batch["audio"]["bytes"]), dtype="float32")
            else:
                # read the clip with soundfile rather than datasets' Audio feature:
                # the Audio feature now requires torchcodec, whose bundled FFmpeg libs
                # don't link on every platform, and we already know these are 16kHz mono
                audio, sr = sf.read(audio_dir / batch["audio"], dtype="float32")
            batch["input_features"] = processor.feature_extractor(
                audio, sampling_rate=sr
            ).input_features[0]
            batch["labels"] = processor.tokenizer(batch["text"]).input_ids
            return batch
        return prepare

    def make_batch_prepare(audio_dir):
        """Batched form of the same work, for set_transform (which hands the
        transform a dict of column -> list and expects the same shape back)."""
        def transform(batch):
            feats, labels = [], []
            for name, text in zip(batch["audio"], batch["text"]):
                if audio_dir is None:
                    audio, sr = sf.read(io.BytesIO(name["bytes"]), dtype="float32")
                else:
                    audio, sr = sf.read(audio_dir / name, dtype="float32")
                feats.append(processor.feature_extractor(
                    audio, sampling_rate=sr).input_features[0])
                labels.append(processor.tokenizer(text).input_ids)
            return {"input_features": feats, "labels": labels}
        return transform

    n_clips = len(ds["train"]) + len(ds["validation"])
    lazy = args.features == "lazy" or (args.features == "auto" and n_clips > args.lazy_threshold)
    if lazy:
        # A large-v3 log-mel is 128x3000 float32 = 1.46 MiB. Precomputing them for
        # a full-corpus run means hundreds of GiB of Arrow cache written before the
        # first step -- 278k clips is ~397 GiB. set_transform computes each batch on
        # demand and caches nothing; at ~0.03s/clip against a ~16s step that is
        # noise. Trainer must not strip the audio/text columns the transform reads,
        # hence remove_unused_columns=False below.
        print(f"features: lazy (on-the-fly, no cache) for {n_clips} clips", file=sys.stderr)
        ds["train"].set_transform(make_batch_prepare(clips_dir))
        ds["validation"].set_transform(make_batch_prepare(eval_clips_dir))
    else:
        # mapped per-split rather than over the whole DatasetDict at once: train and
        # validation can now come from different sources (--eval-dataset, local or
        # Hub), and their manifest columns can differ -- a single remove_columns
        # list across both splits breaks the moment they diverge
        est = n_clips * 128 * 3000 * 4 / 1024 ** 3
        print(f"features: precomputed to cache for {n_clips} clips (~{est:.0f} GiB)",
              file=sys.stderr)
        ds["train"] = ds["train"].map(make_prepare(clips_dir),
                                      remove_columns=ds["train"].column_names,
                                      num_proc=1, desc="extracting log-mel features (train)")
        ds["validation"] = ds["validation"].map(make_prepare(eval_clips_dir),
                                                remove_columns=ds["validation"].column_names,
                                                num_proc=1,
                                                desc="extracting log-mel features (validation)")

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
        # the model places differently is a full word substitution -- the
        # label text attaches punctuation to the preceding word -- which
        # buries the recognition signal this metric is supposed to expose and,
        # because metric_for_best_model="wer", drags checkpoint selection with it.
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
        from peft import LoraConfig, PeftModel, get_peft_model

        # gradient checkpointing freezes the frozen base's input grads, which
        # would stop gradients ever reaching the adapters -- re-enable them
        model.enable_input_require_grads()
        if args.init_adapter:
            print(f"warm-starting LoRA weights from {args.init_adapter} "
                  f"(optimizer/scheduler state not restored -- lr={args.lr})", file=sys.stderr)
            model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        else:
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
        report_to=args.report_to,
        run_name=args.run_name or Path(args.out).name,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        # with lazy features the transform reads the audio/text columns at batch
        # time; Trainer's default column pruning would strip them as "unused"
        # (they are not model inputs) and the transform would raise KeyError
        remove_unused_columns=not lazy,
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

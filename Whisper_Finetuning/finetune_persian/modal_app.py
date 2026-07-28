#!/usr/bin/env python3
"""
Runs the sm_* -> train_whisper -> convert_to_ct2 pipeline on Modal.

One-time setup:
    pip install modal
    modal setup                     # opens a browser to link your Modal account

No secrets are needed for the default run: the HF datasets below are public, as
is the base model repo. Only the legacy `transcribe` stage needs one, and only
if you add NEW links to sm_data/links.jsonl:

    modal secret create speechmatics SM_API_KEY=...
    ENABLE_TRANSCRIBE=1 modal run modal_app.py::transcribe

Data source: the Hugging Face Persian YouTube/podcast collection, NOT the
sm_02 download path. The audio links in sm_data/links.jsonl have rotted --
6 of 14 time out (cdn.imgurl.ir), 6 return 403 (uupload.ir), and the val_a/b/c
transcripts have no entry in links.jsonl at all -- so that path cannot rebuild
a dataset from anywhere, including your own machine. The sm_* stages below are
kept for the day those links are restored; they are not in the default run.

Run everything end-to-end (build dataset from HF -> train LoRA -> convert to
CT2 -> score WER):

    modal run modal_app.py

Or run one stage at a time:

    modal run modal_app.py::build_hf_dataset
    modal run modal_app.py::train
    modal run modal_app.py::convert
    modal run modal_app.py::evaluate

Tune how much audio to pull (per repo, in seconds of kept audio) and which
repos to pull it from:

    modal run modal_app.py --max-seconds-per-repo 7200
    modal run modal_app.py --repos "MohammadGholizadeh/youtube-farsi:transcription:train"

Legacy sm_* stages (need working audio links):

    modal run modal_app.py::seed_data
    modal run modal_app.py::download_audio
    modal run modal_app.py::build_dataset

All generated artifacts (audio, decoded wav, clips, hf_dataset, training runs,
the converted CT2 model) live on a persistent Modal Volume, not in the image,
so re-running a stage picks up where the last one left off. To pull the final
CT2 model down to your machine:

    modal volume get whisper-persian-data models ./models
"""
import os
import pathlib

import modal

LOCAL_DIR = pathlib.Path(__file__).parent
CODE = "/root/finetune_persian"
DATA = "/data"

app = modal.App("whisper-persian-finetune")

# The MohammadGholizadeh Persian YouTube/podcast collection -- the source
# hf_build_dataset.py was written against:
#   https://huggingface.co/collections/MohammadGholizadeh/persian-youtube-asr-datasets
# Format is repo[:text_column[:split]]. youtube-farsi is the only one with a
# `transcription` column and a video_id, so it gets a clean per-video holdout;
# the podcast repos use `sentence`, have a single train split and no id column,
# so hf_build_dataset.py tail-slices them for validation.
#
# Sizes are the reason --max-seconds-per-repo is not optional: Tabaghe16 is
# 132GB of parquet, bplus 90GB, kouman 24GB. The budget stops each stream once
# it has collected that many seconds of KEPT audio.
DEFAULT_HF_REPOS = ",".join([
    "MohammadGholizadeh/youtube-farsi:transcription:train",
    "MohammadGholizadeh/channelbpodcast_dataset_persian:sentence",
    "MohammadGholizadeh/movarekhpodcast_dataset_persian:sentence",
    "MohammadGholizadeh/kouman_dataset_persian:sentence",
    "MohammadGholizadeh/bplus_podcast_persian:sentence",
    "MohammadGholizadeh/Tabaghe16_dataset_persian:sentence",
])

volume = modal.Volume.from_name("whisper-persian-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "requests", "soundfile", "numpy", "datasets", "huggingface_hub",
        "torch", "transformers", "accelerate", "evaluate", "jiwer",
        "peft", "ctranslate2", "faster-whisper",
        # train_whisper.py sets report_to=["tensorboard"]; without this the
        # Trainer raises before the first step
        "tensorboard",
        # torch now ships CUDA 13 (libcublas.so.13), but the ctranslate2 wheel is
        # built against CUDA 12 and dlopens libcublas.so.12 / libcudnn 9 at the
        # first encode -- so faster-whisper loads the model and only then dies
        # with "Library libcublas.so.12 is not found". Different sonames, so the
        # CUDA 12 runtime coexists with torch's CUDA 13 rather than replacing it;
        # _run() puts both on LD_LIBRARY_PATH.
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12>=9",
    )
    .add_local_dir(
        str(LOCAL_DIR),
        remote_path=CODE,
        ignore=["sm_data/audio", "sm_data/dataset", "run*", "models",
                "__pycache__", "*.pyc", ".DS_Store"],
    )
)


def _run(cmd):
    import glob
    import subprocess

    env = dict(os.environ)
    # ctranslate2 (under faster-whisper) dlopens libcublas/libcudnn at the first
    # encode, but the CUDA libs installed as torch's pip deps live under
    # site-packages/nvidia/*/lib, which is not on the loader path. Without this
    # the model loads fine and then dies mid-transcribe with
    # "Library libcublas.so.12 is not found or cannot be loaded".
    try:
        import nvidia

        # a namespace package: __file__ is None, the roots are in __path__
        libs = sorted(
            d
            for base in list(getattr(nvidia, "__path__", []))
            for d in glob.glob(os.path.join(base, "*", "lib"))
        )
        if libs:
            env["LD_LIBRARY_PATH"] = ":".join(
                libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
            )
    except ImportError:
        pass

    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


@app.function(image=image, volumes={DATA: volume}, timeout=600)
def seed_data():
    """Copy the git-committed links.jsonl/urls.txt/transcripts (baked into the
    image) onto the persistent volume, so later stages and any new transcripts
    from `transcribe` all live in one place under /data."""
    import shutil
    from pathlib import Path

    src = Path(CODE) / "sm_data"
    dst = Path(DATA) / "sm_data"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    volume.commit()
    print(f"seeded {dst}")


@app.function(image=image, volumes={DATA: volume}, timeout=3600)
def download_audio(only: str = None):
    cmd = ["python", f"{CODE}/sm_02_download.py",
           "--links", f"{DATA}/sm_data/links.jsonl",
           "--out", f"{DATA}/audio"]
    if only:
        cmd += ["--only", only]
    _run(cmd)
    volume.commit()


# Modal hydrates EVERY registered function when the app starts, so a
# Secret.from_name naming a secret that doesn't exist aborts the whole run --
# including `modal run ...::train`, which never touches this function. Worse,
# the failure usually surfaces as a bare CancelledError, because the concurrent
# image-build tasks get cancelled before the real NotFoundError can print. So
# only attach the secret when you've actually created it and asked for it:
#     modal secret create speechmatics SM_API_KEY=...
#     ENABLE_TRANSCRIBE=1 modal run modal_app.py::transcribe
_transcribe_secrets = (
    [modal.Secret.from_name("speechmatics")]
    if os.environ.get("ENABLE_TRANSCRIBE") == "1"
    else []
)


@app.function(
    image=image,
    volumes={DATA: volume},
    secrets=_transcribe_secrets,
    timeout=7200,
)
def transcribe(operating_point: str = "standard", only: str = None):
    """Only needed if you add new entries to links.jsonl -- the links already
    in this repo already have transcripts under sm_data/transcripts.

    Needs ENABLE_TRANSCRIBE=1 and a `speechmatics` secret; see the note above."""
    if os.environ.get("SM_API_KEY") is None:
        raise SystemExit(
            "SM_API_KEY is not set in the container. Create the secret and re-run with "
            "the flag:\n  modal secret create speechmatics SM_API_KEY=...\n"
            "  ENABLE_TRANSCRIBE=1 modal run modal_app.py::transcribe"
        )
    cmd = ["python", f"{CODE}/sm_03_transcribe.py",
           "--audio", f"{DATA}/audio",
           "--out", f"{DATA}/sm_data/transcripts",
           "--operating-point", operating_point]
    if only:
        cmd += ["--only", only]
    _run(cmd)
    volume.commit()


@app.function(image=image, volumes={DATA: volume}, cpu=8.0, timeout=6 * 3600)
def build_hf_dataset(repos: str = DEFAULT_HF_REPOS, max_seconds_per_repo: float = 3600.0):
    """Build the training set from the Hugging Face collection (the path we
    actually use -- the sm_02 audio links have rotted, see module docstring).

    Streams each repo and stops at the per-repo budget, so this never pulls the
    full 260GB the collection adds up to."""
    cmd = ["python", f"{CODE}/hf_build_dataset.py",
           "--out", f"{DATA}/collection",
           "--max-seconds-per-repo", str(max_seconds_per_repo)]
    for spec in [s.strip() for s in repos.split(",") if s.strip()]:
        cmd += ["--repo", spec]
    _run(cmd)
    volume.commit()


@app.function(image=image, volumes={DATA: volume}, timeout=3600)
def build_dataset():
    cmd = ["python", f"{CODE}/sm_04_build_dataset.py",
           "--audio", f"{DATA}/audio",
           "--transcripts", f"{DATA}/sm_data/transcripts",
           "--out", f"{DATA}/dataset"]
    _run(cmd)
    volume.commit()


@app.function(
    image=image,
    volumes={DATA: volume},
    gpu="A10G",
    timeout=6 * 3600,
)
def train(
    use_lora: bool = True,
    epochs: float = 3.0,
    batch_size: int = 8,
    dataset: str = f"{DATA}/collection/hf_dataset",
    max_eval_samples: int = 256,
    early_stopping_patience: int = 3,
):
    cmd = ["python", f"{CODE}/train_whisper.py",
           "--dataset", dataset,
           "--out", f"{DATA}/run1",
           "--epochs", str(epochs),
           "--batch-size", str(batch_size),
           # eval runs generate() over every clip and can cost more wall clock
           # than the training it measures; the script's own guidance is that
           # 128-256 clips track full-set WER closely enough to pick a checkpoint
           "--max-eval-samples", str(max_eval_samples),
           "--early-stopping-patience", str(early_stopping_patience),
           "--progress", "never"]
    if use_lora:
        cmd.append("--use-lora")
    _run(cmd)
    volume.commit()


# merging the LoRA adapter loads whisper-large-v3 (1.5B params) in fp32 and
# holds base + merged copies at once, so give it real memory rather than
# discovering the OOM after training has already been paid for
@app.function(image=image, volumes={DATA: volume}, cpu=4.0, memory=32768, timeout=3600)
def convert():
    cmd = ["python", f"{CODE}/convert_to_ct2.py",
           f"{DATA}/run1/final",
           f"{DATA}/models/whisper-persian-ct2-int8"]
    _run(cmd)
    volume.commit()


# convert() emits int8_float16, which is the CUDA quantization -- scoring it on
# CPU would make faster-whisper fall back and measure something other than what
# production serves, so eval gets a GPU too (a T4 is plenty for decoding)
@app.function(image=image, volumes={DATA: volume}, gpu="T4", timeout=1800)
def evaluate(dataset: str = f"{DATA}/collection/hf_dataset", limit: int = 500):
    cmd = ["python", f"{CODE}/eval_baseline.py",
           "--model", f"{DATA}/models/whisper-persian-ct2-int8",
           "--dataset", dataset,
           "--split", "validation",
           "--limit", str(limit)]
    _run(cmd)


@app.local_entrypoint()
def main(
    use_lora: bool = True,
    epochs: float = 3.0,
    batch_size: int = 8,
    repos: str = DEFAULT_HF_REPOS,
    max_seconds_per_repo: float = 3600.0,
):
    build_hf_dataset.remote(repos=repos, max_seconds_per_repo=max_seconds_per_repo)
    train.remote(use_lora=use_lora, epochs=epochs, batch_size=batch_size)
    convert.remote()
    evaluate.remote()

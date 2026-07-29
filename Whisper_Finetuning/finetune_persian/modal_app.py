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
SITE = "/usr/local/lib/python3.11/site-packages"

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

_base = (
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
)

_LOCAL = dict(
    remote_path=CODE,
    ignore=["sm_data/audio", "sm_data/dataset", "sm_data/transcripts/val_audio",
            "run*", "models", "__pycache__", "*.pyc", ".DS_Store"],
)


# The CUDA-12 loader path, needed ONLY by ctranslate2/faster-whisper. Keep it
# off every other stage: putting it on the CPU-only dataset builds pulls torch's
# CUDA bindings into a container with no GPU, and the streaming build then dies
# at interpreter shutdown with
#   Fatal Python error: PyGILState_Release: thread state must be current
# -- AFTER writing all its output and printing its summary. The stage reports
# SIGABRT and looks like a build failure when the build actually succeeded.
CUDA_LD_PATH = ":".join([
    f"{SITE}/nvidia/cublas/lib",
    f"{SITE}/nvidia/cudnn/lib",
    f"{SITE}/nvidia/cu13/lib",
])


# add_local_dir must be the LAST step of an image -- Modal rejects a build step
# after it -- so both variants branch from _base and add the code at the end,
# rather than gpu_image deriving from image.
image = _base.add_local_dir(str(LOCAL_DIR), **_LOCAL)

# Same image with the CUDA-12 loader path baked in, for the stages that import
# ctranslate2 IN-PROCESS. LD_LIBRARY_PATH is read by the dynamic loader at
# process start, so setting it inside the function would be too late.
gpu_image = _base.env({"LD_LIBRARY_PATH": CUDA_LD_PATH}).add_local_dir(
    str(LOCAL_DIR), **_LOCAL
)


def _run(cmd, cuda: bool = False):
    """cuda=True adds the CUDA-12 runtime to the subprocess loader path.

    Only ctranslate2 needs it: it dlopens libcublas/libcudnn at the first
    encode, and torch's pip-installed CUDA libs live under
    site-packages/nvidia/*/lib, which is not on the default loader path --
    without it faster-whisper loads the model cleanly and then dies with
    "Library libcublas.so.12 is not found". torch finds its own libs unaided,
    so training does not need this.
    """
    import subprocess

    env = dict(os.environ)
    if cuda:
        env["LD_LIBRARY_PATH"] = ":".join(
            [CUDA_LD_PATH] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )

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


# Per-repo budgets for the ~275h build, in seconds of KEPT audio, sized against
# what each repo actually holds (estimated from the 1h sample: rows x mean clip
# length). movarekh and channelb are taken whole -- their budgets exceed the
# repo. Running one container per repo matters: the build is a sequential
# decode-and-write loop, so six in parallel turns a ~27h job into one bounded by
# its longest single repo.
BIG_BUILD = {
    "MohammadGholizadeh/youtube-farsi:transcription:train": 360000,   # ~100h of ~272h
    "MohammadGholizadeh/Tabaghe16_dataset_persian:sentence": 216000,  # ~60h of ~218h
    "MohammadGholizadeh/bplus_podcast_persian:sentence": 216000,      # ~60h of ~157h
    "MohammadGholizadeh/kouman_dataset_persian:sentence": 108000,     # ~30h of ~41h
    "MohammadGholizadeh/movarekhpodcast_dataset_persian:sentence": 90000,  # all ~23h
    "MohammadGholizadeh/channelbpodcast_dataset_persian:sentence": 90000,  # all ~2h
}


@app.function(image=image, volumes={DATA: volume}, cpu=4.0, timeout=24 * 3600)
def build_one_repo(spec: str, max_seconds: float, out_root: str = f"{DATA}/full"):
    """Build a single repo into its own directory, so the six run concurrently
    and a failure in one does not cost the others their work."""
    tag = spec.split("/")[-1].split(":")[0]
    cmd = ["python", f"{CODE}/hf_build_dataset.py",
           "--repo", spec,
           "--out", f"{out_root}/{tag}",
           "--max-seconds-per-repo", str(max_seconds),
           # see the kouman diagnosis: asterisked stage directions and text too
           # fast to have been spoken are unlearnable, and teach hallucination
           "--drop-annotations",
           "--max-chars-per-sec", "25"]
    _run(cmd)
    volume.commit()
    return tag


@app.function(image=image, volumes={DATA: volume}, cpu=4.0, memory=32768, timeout=6 * 3600)
def blend(out_root: str = f"{DATA}/full", out: str = f"{DATA}/blend"):
    """Merge the per-repo builds into the single dataset train_whisper.py reads.
    Clips are hard-linked rather than copied -- at this scale duplicating them
    is not affordable."""
    from pathlib import Path

    paths = sorted(p for p in Path(out_root).iterdir() if (p / "hf_dataset").is_dir())
    if not paths:
        raise SystemExit(f"no per-repo builds under {out_root}")
    cmd = ["python", f"{CODE}/blend_datasets.py", "--out", out]
    for p in paths:
        cmd += ["--train", f"{p}/hf_dataset", "--val", f"{p}/hf_dataset"]
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


@app.function(image=image, volumes={DATA: volume}, cpu=4.0, timeout=3 * 3600)
def build_val_dataset(out: str = f"{DATA}/val_domain"):
    """Build the real target-domain recordings (val_a/b/c) into a
    validation-ONLY dataset, for use as train_whisper.py --eval-dataset and as
    an eval_baseline.py target.

    These three are the only recordings whose audio survived, so they are worth
    far more as a held-out measure of the actual use case than as a few minutes
    of extra training data. Aggregate WER over public podcasts says nothing
    about how the model handles this audio.

    Two things this has to steer around in sm_04_build_dataset.py:
      - it treats <transcripts>/val as held out, and since these are the only
        recordings with audio, every video would land in validation and it
        exits with "every video is in validation; nothing left to train on".
        So point --transcripts AT the val folder and send --val-dir somewhere
        that does not exist, which puts all three on the val-fraction path.
      - --val-fraction 1.0 then rounds to every video, so the build is pure
        validation with an empty train split, which is what --eval-dataset
        reads.
    """
    cmd = ["python", f"{CODE}/sm_04_build_dataset.py",
           "--audio", f"{DATA}/audio",
           "--transcripts", f"{DATA}/sm_data/transcripts/val",
           "--val-dir", f"{DATA}/__no_such_dir__",
           "--val-fraction", "1.0",
           "--out", out]
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
    eval_steps: int = 200,
    max_train_samples: int = 0,
    out: str = f"{DATA}/run1",
):
    """eval_steps and max_train_samples exist so the whole pipeline can be
    smoke-tested in minutes instead of hours.

    Keep eval_steps BELOW the total step count on a short run. Trainer is
    configured with load_best_model_at_end and metric_for_best_model="wer", so a
    run that finishes without ever reaching an eval has no best checkpoint to
    load and fails at the very end -- after paying for all the training.
    """
    cmd = ["python", f"{CODE}/train_whisper.py",
           "--dataset", dataset,
           "--out", out,
           "--epochs", str(epochs),
           "--batch-size", str(batch_size),
           "--eval-steps", str(eval_steps),
           # eval runs generate() over every clip and can cost more wall clock
           # than the training it measures; the script's own guidance is that
           # 128-256 clips track full-set WER closely enough to pick a checkpoint
           "--max-eval-samples", str(max_eval_samples),
           "--early-stopping-patience", str(early_stopping_patience),
           "--progress", "never"]
    if max_train_samples:
        cmd += ["--max-train-samples", str(max_train_samples)]
    if use_lora:
        cmd.append("--use-lora")
    _run(cmd)
    volume.commit()


# A100-80GB needs a payment method on the Modal account, so this stays on the
# A10G. Note HOW that failed when it was hardcoded: requesting a GPU the
# workspace is not entitled to aborts app HYDRATION, killing every stage
# including the CPU-only dataset build -- the same failure mode as a missing
# Secret. Kept as an env override for if billing is ever set up:
#     BIG_GPU=A100-80GB modal run modal_app.py::big
BIG_GPU = os.environ.get("BIG_GPU", "A10G")


@app.function(image=image, volumes={DATA: volume}, gpu=BIG_GPU, timeout=24 * 3600)
def train_big(epochs: float = 1.0, batch_size: int = 8,
              max_train_samples: int = 80000,
              dataset: str = f"{DATA}/blend/hf_dataset"):
    """Train on the blended corpus, sized to what an A10G can finish.

    The A10G measured 7.6 s/step at batch 8 / grad-accum 2, so the GPU budget --
    not the amount of data on disk -- is the binding constraint: 80k clips for
    one epoch is ~5000 steps, ~11h. Training the whole ~275h build would be
    ~40h at one epoch, past the 24h function timeout.

    So the build keeps everything and this samples from it. --max-train-samples
    shuffles with the seed before selecting, so the subset is a real sample
    across all six sources rather than whichever source sorts first, and it is
    stable across runs for comparability.

    --features stays on auto: above the 20k-clip threshold it picks lazy, which
    is essential here -- precomputing log-mels for a corpus this size would want
    hundreds of GiB of Arrow cache before the first step.
    """
    cmd = ["python", f"{CODE}/train_whisper.py",
           "--dataset", dataset,
           "--out", f"{DATA}/run_big",
           "--epochs", str(epochs),
           "--batch-size", str(batch_size),
           "--grad-accum", "2",
           "--use-lora",
           "--max-eval-samples", "256",
           "--early-stopping-patience", "3",
           "--eval-steps", "500",
           # lazy features read the audio/text columns per batch; a couple of
           # workers keep extraction off the training thread's critical path
           "--dataloader-workers", "4",
           "--progress", "never"]
    if max_train_samples:
        cmd += ["--max-train-samples", str(max_train_samples)]
    _run(cmd)
    volume.commit()


# merging the LoRA adapter loads whisper-large-v3 (1.5B params) in fp32 and
# holds base + merged copies at once, so give it real memory rather than
# discovering the OOM after training has already been paid for
@app.function(image=image, volumes={DATA: volume}, cpu=4.0, memory=32768, timeout=3600)
def convert():
    # --force so the stage is re-runnable: the converter refuses to write into an
    # existing output dir, which otherwise makes every convert after the first
    # one fail -- and it fails AFTER redoing the LoRA merge. The output is
    # regenerable from run1/final, so overwriting is the right default here.
    cmd = ["python", f"{CODE}/convert_to_ct2.py",
           f"{DATA}/run1/final",
           f"{DATA}/models/whisper-persian-ct2-int8",
           "--force"]
    _run(cmd)
    volume.commit()


# convert() emits int8_float16, which is the CUDA quantization -- scoring it on
# CPU would make faster-whisper fall back and measure something other than what
# production serves, so eval gets a GPU too (a T4 is plenty for decoding)
# 2h, not the 1800s this started at: decoding runs ~2 s/clip on a T4, so a
# full 959-clip held-out set needs ~32 min and the old cap killed it at 850 --
# after 28 minutes of GPU time, with no report written.
@app.function(image=gpu_image, volumes={DATA: volume}, gpu="T4", timeout=2 * 3600)
def evaluate(dataset: str = f"{DATA}/collection/hf_dataset", limit: int = 500):
    cmd = ["python", f"{CODE}/eval_baseline.py",
           "--model", f"{DATA}/models/whisper-persian-ct2-int8",
           "--dataset", dataset,
           "--split", "validation",
           "--limit", str(limit)]
    _run(cmd, cuda=True)


@app.function(image=gpu_image, volumes={DATA: volume}, gpu="T4", timeout=1800)
def diagnose_source(source: str = "kouman_dataset_persian", n: int = 10,
                    dataset: str = f"{DATA}/collection/hf_dataset"):
    """Why does one source score far worse than its peers? Three causes look
    identical in an aggregate WER and call for opposite responses:

      misaligned audio/text -> hyp is fluent Persian but unrelated to ref
      bad/machine transcripts -> ref itself is garbled or truncated
      genuinely hard audio -> hyp tracks ref but is noisy

    chars-per-second separates the first two from the third before any decoding:
    a source whose text does not match its audio has a ratio unlike every other
    source, because the text was never a transcript of that audio.
    """
    import statistics
    from pathlib import Path

    from datasets import load_from_disk

    ds = load_from_disk(dataset)
    rows = [r for split in ds for r in ds[split]]

    by_source = {}
    for r in rows:
        by_source.setdefault(r.get("source") or "?", []).append(r)

    print(f"{'source':32s} {'clips':>6s} {'hours':>7s} {'sec/clip':>9s} "
          f"{'chars':>7s} {'chars/sec':>10s}")
    for src in sorted(by_source):
        rs = by_source[src]
        durs = [r["duration"] for r in rs]
        chars = [len(r["text"]) for r in rs]
        cps = [c / d for c, d in zip(chars, durs) if d > 0]
        print(f"{src:32s} {len(rs):6d} {sum(durs) / 3600:7.2f} "
              f"{statistics.mean(durs):9.2f} {statistics.mean(chars):7.1f} "
              f"{statistics.mean(cps):10.2f}")

    target = by_source.get(source)
    if not target:
        print(f"\nno rows for source={source}; have {sorted(by_source)}")
        return

    print(f"\n--- {source}: ref vs hyp on {n} clips ---")
    from faster_whisper import WhisperModel

    model = WhisperModel(f"{DATA}/models/whisper-persian-ct2-int8",
                         device="cuda", compute_type="int8_float16")
    clips = Path(dataset).parent / "clips"
    for r in target[:n]:
        segments, _ = model.transcribe(str(clips / r["audio"]), language="fa",
                                       beam_size=5, condition_on_previous_text=False)
        hyp = " ".join(s.text for s in segments).strip()
        print(f"\n  [{r['audio']}]  {r['duration']:.1f}s  "
              f"{len(r['text']) / max(r['duration'], 0.01):.1f} chars/s")
        print(f"  ref: {r['text']}")
        print(f"  hyp: {hyp}")


@app.function(image=image, volumes={DATA: volume}, timeout=900)
def text_hygiene(dataset: str = f"{DATA}/collection/hf_dataset"):
    """How much of each source is non-speech annotation rather than transcript?

    Bracketed stage directions (* applause *, on-screen text) are unlearnable --
    the words are not in the audio -- so every such row is ~100% WER no matter
    how good the model gets, and training on them teaches hallucination.
    """
    import re
    import statistics

    from datasets import load_from_disk

    ds = load_from_disk(dataset)
    rows = [r for split in ds for r in ds[split]]
    by_source = {}
    for r in rows:
        by_source.setdefault(r.get("source") or "?", []).append(r)

    star = re.compile(r"[*\[\]()（）【】]")
    print(f"{'source':32s} {'clips':>6s} {'annot%':>7s} {'cps p50':>8s} "
          f"{'cps p95':>8s} {'>20cps%':>8s}")
    for src in sorted(by_source):
        rs = by_source[src]
        cps = sorted(len(r["text"]) / max(r["duration"], 0.01) for r in rs)
        n_star = sum(1 for r in rs if star.search(r["text"]))
        n_fast = sum(1 for c in cps if c > 20)
        print(f"{src:32s} {len(rs):6d} {100 * n_star / len(rs):6.1f}% "
              f"{statistics.median(cps):8.2f} {cps[int(0.95 * len(cps)) - 1]:8.2f} "
              f"{100 * n_fast / len(rs):7.1f}%")

    print("\n--- sample annotation rows (any source) ---")
    shown = 0
    for r in rows:
        if star.search(r["text"]) and shown < 8:
            print(f"  [{r.get('source')}] {r['duration']:.1f}s  {r['text'][:90]}")
            shown += 1


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


@app.local_entrypoint()
def build_all():
    """Phase 1 only: the six per-repo builds in parallel, then the blend.

    Split from training deliberately -- this phase is hours long, and with
    `modal run --detach` only the LAST triggered function survives a dropped
    local connection, so a single build->blend->train entrypoint can lose the
    chain part-way. Each phase is separately restartable, and the volume keeps
    whatever already finished.
    """
    tags = list(build_one_repo.starmap(
        [(spec, budget) for spec, budget in BIG_BUILD.items()]
    ))
    print("built:", tags)
    blend.remote()
    print("blended -> /data/blend  (now: modal run --detach modal_app.py::train_big)")


@app.local_entrypoint()
def big(epochs: float = 1.0, batch_size: int = 8, max_train_samples: int = 80000):
    """The ~275h build: six repos in parallel, blended, then trained on the
    A10G over a seed-stable sample of the result (see train_big for why the GPU
    budget, not the disk, sets that number). One epoch rather than three -- on a
    corpus this size the extra passes mostly overfit and each costs ~11h."""
    tags = list(build_one_repo.starmap(
        [(spec, budget) for spec, budget in BIG_BUILD.items()]
    ))
    print("built:", tags)
    blend.remote()
    train_big.remote(epochs=epochs, batch_size=batch_size,
                     max_train_samples=max_train_samples)
    convert.remote()
    evaluate.remote(dataset=f"{DATA}/blend/hf_dataset")

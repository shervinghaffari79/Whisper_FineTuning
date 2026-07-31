# Persian Whisper fine-tuning on Modal

Fine-tunes `AmirMohseni/whisper-large-v3-persian-bf16` with LoRA and converts the
result to the CTranslate2 int8 format `asr_engine.py` serves in production --
that app lives in its own repo, [STT_Persian](https://github.com/shervinghaffari79/STT_Persian).

Everything runs on [Modal](https://modal.com) (serverless GPU). You need no local
GPU, and no data on your machine — the pipeline pulls its audio from public
Hugging Face datasets.

`modal_app.py` is a thin orchestration layer: every stage shells out to the same
scripts you would run locally, so nothing here is Modal-specific except where to
run it and on what hardware.

---

## Quick start

```bash
pip install modal
modal setup          # opens a browser to link your Modal account
```

```bash
cd finetune_persian
modal run --detach modal_app.py
```

That runs the whole thing: build a ~12h dataset → LoRA fine-tune on an A10G →
merge and convert to CT2 → score WER. Expect **~6–7 hours**, most of it training
(~5.6h train+eval on the A10G alone, at ~7.6 s/step).

No secrets are required. The datasets and the base model are public.

Download the finished model:

```bash
modal volume get whisper-persian-data models/whisper-persian-ct2-int8 ./models
export CT2_MODEL_DIR=$PWD/models/whisper-persian-ct2-int8
```

`asr_engine.py` reads `CT2_MODEL_DIR` — no code change needed to serve it.

---

## Results from the reference run

From before `--max-seconds-per-repo`'s default doubled from 3600s (1h/repo) to
7200s (2h/repo) -- these numbers are from the smaller build and haven't been
re-measured at the new default yet. 6,748 clips / 6.0h of audio, LoRA, 3
epochs, ~2.4h on an A10G:

| Stage | WER |
|---|---|
| eval @ step 200 | 31.85% |
| eval @ step 600 | 27.28% |
| eval @ step 1140 (final) | 25.22% |
| **converted CT2 model, 500 held-out clips** | **23.35%** (CER 12.53%) |

Per-source breakdown of that final number:

| Source | Clips | WER | CER |
|---|---:|---:|---:|
| `bplus_podcast` | 83 | 9.45% | 3.49% |
| `movarekhpodcast` | 90 | 10.69% | 2.85% |
| `channelbpodcast` | 117 | 18.80% | 7.96% |
| `Tabaghe16` | 54 | 20.05% | 9.97% |
| `youtube-farsi` | 31 | 23.26% | 11.51% |
| `kouman` | 125 | 51.10% | 33.93% |

**Run-to-run variance is real.** Two evals of the same model over the same
seed-fixed subset gave 23.35%/12.53% and 23.65%/13.75%. That spread is
non-determinism in `int8_float16` GPU inference. Treat results as ~23.5% ± a few
tenths; don't read a 0.3pp "improvement" as signal.

---

## Stages

Each is independently runnable — `modal run modal_app.py::<stage>`. All output
lands on the persistent volume `whisper-persian-data`, so a stage never redoes
the previous one's work.

| Stage | Hardware | Writes | Notes |
|---|---|---|---|
| `build_hf_dataset` | 8 CPU | `/data/collection` | streams HF repos, ~2h audio each |
| `train` | A10G | `/data/run1` | LoRA, 3 epochs |
| `convert` | 4 CPU / 32GB | `/data/models/...` | merges adapter, quantizes |
| `evaluate` | T4 | — | WER/CER + per-source table |
| `diagnose_source` | T4 | — | why is one source scoring badly |
| `text_hygiene` | CPU | — | annotation/chars-per-sec audit |

Scaling up (see [Scaling](#scaling) below):

| Stage | Hardware | Writes |
|---|---|---|
| `build_one_repo` | 4 CPU | `/data/full/<tag>` |
| `blend` | 4 CPU / 32GB | `/data/blend` |
| `train_big` | A10G | `/data/run_big` |

### Local entrypoints

The pre-wired chains — `modal run --detach modal_app.py::<name>`:

| Entrypoint | Chains | Use for |
|---|---|---|
| `main` | `build_hf_dataset → train → convert → evaluate` | the default run (public HF data only) |
| `combined` | `build_hf_dataset → train (own + public, separate eval reports) → convert → evaluate ×2` | training on your own data too — see [Training on both at once](#training-on-both-at-once) |
| `build_all` | six `build_one_repo` in parallel `→ blend` | phase 1 of scaling up, split out since `--detach` only survives the *last* triggered function |
| `big` | `build_all`'s chain `→ train_big → convert → evaluate` | the full ~275h corpus in one call — see [Scaling](#scaling) |

Useful flags:

```bash
# more audio per repo (default 7200s = 2h of kept audio each)
modal run modal_app.py --max-seconds-per-repo 14400

# one repo only
modal run modal_app.py --repos "MohammadGholizadeh/youtube-farsi:transcription:train"
```

---

## Where the data comes from

Two sources, both consumed the same way by `train_whisper.py` (a local disk
path or a Hugging Face Hub repo id — see [Local / non-Modal use](#local--non-modal-use)):

- **Public Persian YouTube/podcast corpora** — the
  [MohammadGholizadeh collection](https://huggingface.co/collections/MohammadGholizadeh/persian-youtube-asr-datasets),
  which is what `hf_build_dataset.py` was written against. Six repos, ~713h total,
  ~260GB of parquet. It is **streamed**, and `--max-seconds-per-repo` stops each
  stream once it has enough — without that budget you would pull all 260GB.

  Only `youtube-farsi` has a `video_id`, so it gets a proper held-out-by-recording
  validation split. The podcast repos are a single `train` split with no recording
  id, so they are tail-sliced — same speaker on both sides, which is weaker but is
  what is available without re-diarizing.

- **Your own recordings**, pushed to the Hub with `push_to_hub.py` (see its
  module docstring) and passed straight to `train()`/`evaluate()` via
  `--dataset <repo id>` — e.g. `shervingh2000/behpardaz`.

### Training on both at once

```bash
modal run --detach modal_app.py::combined
```

Concatenates your own dataset's train split onto the public HF build's, while
validation tracking during training stays on your own dataset (the real
target-domain data) — not a blend of both, so checkpoint selection and
early stopping keep optimizing for the metric that actually matters. After
training, it reports WER/CER **separately** for each validation set (own,
then public), since a blended number isn't comparable run to run as either
source's size changes.

```bash
# override the defaults
modal run --detach modal_app.py::combined --own-dataset shervingh2000/behpardaz \
    --max-seconds-per-repo 14400 --epochs 2
```

Calling `train()` directly gives the same building blocks individually:
`--dataset` (what to train on), `--extra-dataset` (a second dataset whose
train split gets concatenated on), `--eval-dataset` (which validation set to
track live). Any of the three can be a local disk path or a Hub repo id, in
any combination — see `train_whisper.py`'s `--extra-dataset` docstring for
how heterogeneous sources (local filenames vs. Hub-embedded audio) get
reconciled before concatenation.

### Data quality: `kouman`

`kouman` scores 51% WER against 9–23% for every other source. It is **not**
misaligned — most of its refs transcribe almost exactly. The damage is:

- **11.7%** of rows are non-speech stage directions in asterisks
  (`* applause *`, on-screen text). Those words are not in the audio, so they
  are ~100% WER however good the model gets, and training on them teaches the
  model to invent text nobody said.
- **24%** of rows exceed 20 chars/second — faster than the clip is long, i.e.
  on-screen text again.

`hf_build_dataset.py --drop-annotations --max-chars-per-sec 25` removes both
(measured on kouman: 18/114 rows dropped as annotation, 1 as too fast, 95 kept).
The scaling stages pass these by default.

Asterisks are a safe marker — they appear nowhere else in these transcripts.
**Brackets deliberately are not filtered**: `youtube-farsi` uses `[میزبان:]` for
speaker labels (noise) but `(BPM)` for spoken parentheticals (not noise).

---

## Scaling

The reference run uses 2h per repo. To go bigger:

```bash
modal run --detach modal_app.py::build_all      # ~275h, 6 repos in parallel, then blend
modal run --detach modal_app.py::train_big      # trains on a sample of the blend
```

`build_all` runs one container per repo. The build is a sequential
decode-and-write loop, so running six concurrently turns a ~27h job into one
bounded by its longest single repo.

**GPU time, not disk, is the binding constraint.** The A10G measures ~7.6 s/step
at batch 8 / grad-accum 2:

| Clips | Epochs | Steps | Approx A10G time |
|---:|---:|---:|---|
| 80,000 | 1 | ~5,000 | ~11h |
| 300,000 (full ~275h build) | 1 | ~19,000 | ~40h — **over the 24h function timeout** |

So `train_big` defaults to `--max-train-samples 80000`. That shuffles with the
seed *before* selecting, so it is a real sample across all six sources and is
stable between runs. The rest of the corpus stays on the volume for later runs.

A bigger GPU needs a payment method on the Modal account:

```bash
BIG_GPU=A100-80GB modal run --detach modal_app.py::train_big --batch-size 24
```

If you do move to an A100/H100, note that `train_whisper.py` hardcodes
`fp16=torch.cuda.is_available()`, and its comment explains why: a T4 has no
native bf16. On an A100/H100 that reasoning inverts — bf16 is native and the base
checkpoint is already bf16 — so getting full value from the card means editing
that line, not just changing `gpu=`.

---

## Gotchas

Each of these cost a run. They are all in `modal_app.py` with comments; this is
the short version.

**A missing resource aborts the entire app, not just the stage that needs it.**
Modal hydrates *every* registered function at startup. A `Secret.from_name` for a
secret you never created, or a `gpu=` your workspace is not entitled to, kills
`modal run ...::train` too — a stage that never touches either. Worse, it usually
surfaces as a bare `CancelledError`, because the concurrent image build gets
cancelled before the real `NotFoundError` can print. If you get an instant
`CancelledError` with no useful message, suspect this first, and bisect by
running a trivial function.

Concretely: **`train()` always has the `wandb` secret attached**, even if
`--report-to` never includes wandb — so that secret must exist before *any*
call to `modal_app.py::train` succeeds:

```bash
modal secret create wandb WANDB_API_KEY=...
```

(It isn't gated behind a flag on purpose — toggling a function's secrets list
between `modal run` invocations of the same app hits a different failure,
"Function has N dependencies but container got N+1 object ids"; see the
`_wandb_secrets` comment in `modal_app.py`.) A bigger `BIG_GPU=` is opt-in the
normal way, gated behind that env var.

**`ctranslate2` needs CUDA 12; torch now ships CUDA 13.** `libcublas.so.13`
exists, `libcublas.so.12` is what gets dlopened — at the *first encode*, long
after the model loads cleanly. Both runtimes are installed side by side and put
on `LD_LIBRARY_PATH`. That variable is read by the dynamic loader at process
start, so it is set on the image (not just in `_run`) for anything importing
`ctranslate2` in-process.

**`train_whisper.py` requires `tensorboard`** via `report_to=["tensorboard"]`, or
the Trainer raises before step 1.

**`--detach` keeps only the *last* triggered function alive.** A single
entrypoint chaining build → blend → train can lose the tail when the local
process goes away, which is why `build_all` and `train_big` are separate.

**Training progress does not stream reliably.** Output buffers through the
detached CLI and can look stalled for an hour while everything is fine. Check the
volume instead:

```bash
modal volume ls whisper-persian-data run1       # checkpoints appearing = progress
modal app list                                  # task count > 0 = still running
```

To read live metrics, pull the newest checkpoint's `trainer_state.json` — its
`log_history` has every loss and eval WER.

**Beware over-filtering your own log output.** A `grep -E "WER|CER|clips"` over
the eval output passes the per-source *header* (it contains those words) and
drops every data row beneath it, which looks exactly like a bug in the eval
script. It is not.

---

## Costs

Rough shape of the reference run (check
[Modal's pricing](https://modal.com/pricing) for current rates):

- image build: ~80s, once, then cached
- dataset build (12h audio): ~45–60 min on CPU
- training: ~4.9h + ~0.7h eval overhead on one A10G
- convert + evaluate: ~20–30 min

The free tier covers the A10G. A100/H100 requires a payment method.

---

## Local / non-Modal use

Every script runs standalone; `modal_app.py` only chooses where. See each
script's module docstring for its flags. The pipeline order is:

```
hf_build_dataset (Hugging Face sources) ─┐
                                          ├→ train_whisper → convert_to_ct2 → eval_baseline
push_to_hub (your own recordings) ───────┘      ↑
                          blend_datasets merges multiple local builds
```

`train_whisper.py --dataset`, `--eval-dataset`, and `--extra-dataset` each
accept either a local disk path (a DatasetDict built by `hf_build_dataset.py`,
with a sibling `clips/` folder) or a Hugging Face Hub dataset repo id (audio
embedded, no clips dir needed) — e.g. `--dataset shervingh2000/behpardaz`.
`--extra-dataset` concatenates a second source's train split onto
`--dataset`'s for training on both at once, while `--eval-dataset` still
controls which single validation set gets tracked live (see
[Training on both at once](#training-on-both-at-once)).

Diagnostics: `report_run.py` (training curves), `peek_checkpoint.py` (mid-run
sanity check), `diagnose_ct2.py` (inspect a converted model).

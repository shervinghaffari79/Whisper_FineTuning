#!/usr/bin/env python3
"""
Report the training/evaluation curves for a run: train loss, eval loss, WER,
raw WER, learning rate, and a per-eval table.

Reads trainer_state.json rather than the TensorBoard event files -- Trainer
writes the full log_history there at every checkpoint save, it needs no extra
dependency, and it survives the run being interrupted (the last checkpoint
still has everything up to that point).

Emits three things, in decreasing order of what a machine can be missing:
  - a markdown table on stdout, always
  - metrics.csv, always
  - curves.png, only if matplotlib is installed (it is not a dependency of
    anything else here, so a missing matplotlib degrades the report rather
    than failing the run)

Usage:
    python report_run.py --run ./run1
    python report_run.py --run ./run2 --out ./run2/report
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def find_state(run_dir: Path) -> Path:
    """Prefer the highest-numbered checkpoint: load_best_model_at_end rewinds
    the MODEL to the best step, but trainer_state.json in the newest checkpoint
    still holds the complete log_history for the whole run."""
    direct = run_dir / "trainer_state.json"
    checkpoints = sorted(
        (p for p in run_dir.glob("checkpoint-*") if (p / "trainer_state.json").is_file()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if checkpoints:
        return checkpoints[-1] / "trainer_state.json"
    if direct.is_file():
        return direct
    sys.exit(f"no trainer_state.json under {run_dir} -- looked in {run_dir}/ and "
             f"{run_dir}/checkpoint-*/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="./run1", help="training output dir (--out of train_whisper.py)")
    parser.add_argument("--state", default=None, help="explicit trainer_state.json path")
    parser.add_argument("--out", default=None, help="report dir (default: <run>/report)")
    parser.add_argument("--no-plot", action="store_true", help="skip curves.png even if matplotlib is present")
    args = parser.parse_args()

    run_dir = Path(args.run)
    state_path = Path(args.state) if args.state else find_state(run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    if not history:
        sys.exit(f"{state_path} has an empty log_history")

    # Trainer emits train logs and eval logs as SEPARATE entries; an eval entry
    # has no "loss" key and a train entry has no "eval_loss". Splitting on that
    # rather than on epoch keeps the two series independent, which matters
    # because they are logged on different intervals (logging_steps vs eval_steps).
    train_pts = [h for h in history if "loss" in h and "eval_loss" not in h]
    eval_pts = [h for h in history if "eval_loss" in h]

    print(f"state   : {state_path}")
    print(f"logged  : {len(train_pts)} train points, {len(eval_pts)} eval points")
    if state.get("best_metric") is not None:
        print(f"best    : {state['best_metric']:.2f} WER @ {state.get('best_model_checkpoint')}")

    if eval_pts:
        print("\n| step | epoch | eval_loss | wer | wer_raw |")
        print("|---:|---:|---:|---:|---:|")
        for e in eval_pts:
            wer = e.get("eval_wer")
            raw = e.get("eval_wer_raw")
            print(f"| {e.get('step', '')} | {e.get('epoch', 0):.2f} | {e['eval_loss']:.4f} | "
                  f"{wer:.2f} | {raw:.2f} |" if wer is not None and raw is not None
                  else f"| {e.get('step', '')} | {e.get('epoch', 0):.2f} | {e['eval_loss']:.4f} | - | - |")

    csv_path = out_dir / "metrics.csv"
    fields = ["step", "epoch", "loss", "grad_norm", "learning_rate",
              "eval_loss", "eval_wer", "eval_wer_raw", "eval_runtime"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for h in sorted(history, key=lambda h: (h.get("step", 0), "eval_loss" in h)):
            writer.writerow(h)
    print(f"\nmetrics -> {csv_path}")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display on a headless server
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping curves.png "
              "(pip install matplotlib to get it)", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot([h["step"] for h in train_pts], [h["loss"] for h in train_pts],
            label="train loss", color="tab:blue", alpha=0.8)
    if eval_pts:
        ax.plot([h["step"] for h in eval_pts], [h["eval_loss"] for h in eval_pts],
                label="eval loss", color="tab:orange", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    wer_pts = [h for h in eval_pts if h.get("eval_wer") is not None]
    if wer_pts:
        ax.plot([h["step"] for h in wer_pts], [h["eval_wer"] for h in wer_pts],
                label="WER (normalized)", color="tab:green", marker="o")
        raw_pts = [h for h in wer_pts if h.get("eval_wer_raw") is not None]
        if raw_pts:
            ax.plot([h["step"] for h in raw_pts], [h["eval_wer_raw"] for h in raw_pts],
                    label="WER (raw)", color="tab:red", marker="s", alpha=0.6)
        best = min(wer_pts, key=lambda h: h["eval_wer"])
        ax.axvline(best["step"], color="grey", linestyle="--", alpha=0.7)
        ax.annotate(f"best {best['eval_wer']:.2f} @ {best['step']}",
                    xy=(best["step"], best["eval_wer"]), xytext=(5, 10),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel("WER %")
    ax.set_title("Word error rate")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    lr_pts = [h for h in train_pts if h.get("learning_rate") is not None]
    ax.plot([h["step"] for h in lr_pts], [h["learning_rate"] for h in lr_pts],
            color="tab:purple")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedule")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{run_dir.name} -- {len(train_pts)} train / {len(eval_pts)} eval points")
    fig.tight_layout()
    png_path = out_dir / "curves.png"
    fig.savefig(png_path, dpi=130)
    print(f"curves  -> {png_path}")


if __name__ == "__main__":
    main()

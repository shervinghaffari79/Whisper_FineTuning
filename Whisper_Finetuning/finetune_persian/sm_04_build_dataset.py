#!/usr/bin/env python3
"""
Step 4 -- turn the Speechmatics word-level output + downloaded audio into a
Whisper fine-tuning dataset (same on-disk shape sm/train_whisper.py expects:
an HF dataset of {audio: filename, text: str} plus a sibling clips/ folder).

Why segment on word timings instead of reusing caption cues: Speechmatics
returns a start/end per word, so clips can be cut in the silence BETWEEN
words. The YouTube-caption path had to trust approximate cue timings and
routinely clipped word onsets, which teaches the model to hallucinate the
missing syllable.

Segmentation rule: accumulate words until a pause long enough to be a natural
boundary (--split-gap) once the segment is at least --min-sec, or until
--max-sec forces a cut. Whisper's receptive field is 30s, so --max-sec stays
safely under it.

Usage:
    python sm_04_build_dataset.py --audio ./sm_data/audio \
        --transcripts ./sm_data/transcripts --out ./sm_data/dataset
"""
import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import soundfile as sf
from datasets import Dataset, DatasetDict

SAMPLE_RATE = 16000
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه"})
_DIACRITICS = re.compile(r"[ً-ْـ]")


def normalize_fa(text: str) -> str:
    """Light normalization only -- keep punctuation and ZWNJ, which are real
    orthography the model should learn to produce. Unify the Arabic/Persian
    codepoint variants that are genuinely interchangeable."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _DIACRITICS.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def decode_to_wav(src: Path, dest: Path) -> Path:
    """Decode the source m4a to 16kHz mono wav once, so clip slicing is a cheap
    array slice rather than one ffmpeg spawn per clip."""
    if dest.exists():
        return dest
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-ar", str(SAMPLE_RATE), "-ac", "1", str(dest)],
        check=True,
    )
    return dest


def iter_words(sm_json: dict):
    """Yield (start, end, content, confidence) for word results, attaching any
    punctuation to the preceding word so text reads naturally."""
    pending = None
    for res in sm_json.get("results", []):
        alts = res.get("alternatives") or [{}]
        content = alts[0].get("content", "")
        if not content:
            continue
        if res.get("type") == "punctuation":
            if pending:
                pending[2] += content
            continue
        if pending:
            yield tuple(pending)
        pending = [res.get("start_time", 0.0), res.get("end_time", 0.0),
                   content, alts[0].get("confidence", 1.0)]
    if pending:
        yield tuple(pending)


def segment(words, min_sec, max_sec, split_gap):
    """Group words into utterance-sized segments broken at natural pauses."""
    segments, cur = [], []
    for w in words:
        if cur:
            gap = w[0] - cur[-1][1]
            dur = cur[-1][1] - cur[0][0]
            if (gap >= split_gap and dur >= min_sec) or (w[1] - cur[0][0]) > max_sec:
                segments.append(cur)
                cur = []
        cur.append(w)
    if cur:
        segments.append(cur)
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="./sm_data/audio")
    parser.add_argument("--transcripts", default="./sm_data/transcripts")
    parser.add_argument("--out", default="./sm_data/dataset")
    parser.add_argument("--min-sec", type=float, default=4.0)
    parser.add_argument("--max-sec", type=float, default=26.0)
    parser.add_argument("--split-gap", type=float, default=0.45,
                        help="pause (s) that may end a segment")
    parser.add_argument("--pad", type=float, default=0.10,
                        help="seconds of context kept on each side of a clip")
    parser.add_argument("--min-confidence", type=float, default=0.60,
                        help="drop segments whose mean word confidence is below this")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--only", default=None, help="build from just this video id (stress test)")
    args = parser.parse_args()

    audio_dir, tr_dir, out_dir = Path(args.audio), Path(args.transcripts), Path(args.out)
    clips_dir = out_dir / "clips"
    wav_dir = out_dir / "wav16k"
    clips_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    transcripts = sorted(tr_dir.glob("*.json"))
    if args.only:
        transcripts = [t for t in transcripts if t.stem == args.only]
        if not transcripts:
            sys.exit(f"no transcript for {args.only} in {tr_dir}")

    rows, stats = [], []
    for tr_path in transcripts:
        vid = tr_path.stem
        src = audio_dir / f"{vid}.m4a"
        if not src.exists():
            print(f"[{vid}] no audio file, skipping", file=sys.stderr)
            continue
        print(f"[{vid}] decoding...", file=sys.stderr)
        wav_path = decode_to_wav(src, wav_dir / f"{vid}.wav")
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != SAMPLE_RATE:
            sys.exit(f"{wav_path}: expected {SAMPLE_RATE}Hz, got {sr}")

        sm_json = json.loads(tr_path.read_text(encoding="utf-8"))
        words = list(iter_words(sm_json))
        segs = segment(words, args.min_sec, args.max_sec, args.split_gap)

        kept = 0
        for i, seg in enumerate(segs):
            if len(seg) < args.min_words:
                continue
            mean_conf = sum(w[3] for w in seg) / len(seg)
            if mean_conf < args.min_confidence:
                continue
            text = normalize_fa(" ".join(w[2] for w in seg))
            if not text:
                continue
            start = max(0.0, seg[0][0] - args.pad)
            end = min(len(audio) / SAMPLE_RATE, seg[-1][1] + args.pad)
            if end - start < 1.0 or end - start > args.max_sec + 2 * args.pad + 1:
                continue
            clip = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            name = f"{vid}_{i:05d}.wav"
            sf.write(clips_dir / name, clip, SAMPLE_RATE)
            rows.append({"audio": name, "text": text, "video_id": vid,
                         "start": round(start, 3), "duration": round(end - start, 3),
                         "confidence": round(mean_conf, 4)})
            kept += 1
        total_h = sum(r["duration"] for r in rows if r["video_id"] == vid) / 3600
        stats.append((vid, len(words), len(segs), kept, total_h))
        print(f"[{vid}] {len(words)} words -> {len(segs)} segments -> {kept} clips "
              f"({total_h:.2f}h)", file=sys.stderr)

    if not rows:
        sys.exit("no clips produced")

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # split by video so train/val never share audio from the same recording
    videos = sorted({r["video_id"] for r in rows})
    n_val = max(1, round(len(videos) * args.val_fraction)) if len(videos) > 1 else 0
    val_videos = set(videos[:n_val])
    train_rows = [r for r in rows if r["video_id"] not in val_videos]
    val_rows = [r for r in rows if r["video_id"] in val_videos]
    if not val_rows:  # single-video (stress test): carve a tail slice
        cut = max(1, len(train_rows) // 10)
        val_rows, train_rows = train_rows[-cut:], train_rows[:-cut]

    DatasetDict({
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(val_rows),
    }).save_to_disk(str(out_dir / "hf_dataset"))

    total_h = sum(r["duration"] for r in rows) / 3600
    print()
    print(f"{'video':16s} {'words':>7s} {'segs':>6s} {'clips':>6s} {'hours':>7s}")
    for vid, nw, ns, nk, h in stats:
        print(f"{vid:16s} {nw:7d} {ns:6d} {nk:6d} {h:7.2f}")
    print(f"\ntotal: {len(rows)} clips, {total_h:.2f} hours "
          f"(train={len(train_rows)}, val={len(val_rows)})")
    print(f"dataset  -> {out_dir / 'hf_dataset'}")
    print(f"clips    -> {clips_dir}")


if __name__ == "__main__":
    main()

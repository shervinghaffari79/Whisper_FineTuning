#!/usr/bin/env python3
"""
Step 3b -- alternative to sm_03 for audio you ALREADY have a transcript for.
Converts .srt / youtube_transcript_api .json into the Speechmatics json-v2
shape sm_04 consumes, so the rest of the pipeline is unchanged.

Run it INSTEAD of sm_03 for those ids (sm_03 skips any id whose transcript
already exists, so the two can be mixed in one sm_data/transcripts dir).

Timing granularity is the whole point of this script:

  --granularity cue (default)
      One result entry per subtitle cue, carrying the cue's real start/end.
      sm_04 then groups whole cues and every clip boundary falls where the
      caption author put one. Segment lengths are whatever the captions used.

  --granularity word
      Splits each cue on whitespace and distributes the cue's span across the
      words in proportion to their length. The timings are INVENTED. sm_04's
      docstring exists because approximate cue timings clip word onsets and
      teach the model to hallucinate the missing syllable -- this mode has the
      same failure at finer scale. Use it only if you know the captions are
      tightly timed, and pair it with a larger --pad in sm_04.

For real word timings from plain text, run a forced aligner (WhisperX,
MMS, ctc-segmentation) over audio+text and emit its output instead.

Usage:
    python sm_03b_import_transcripts.py --src ./sm_data/captions \
        --out ./sm_data/transcripts
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Caption files carry non-speech annotations the model must never learn to
# emit: [music], (laughs), * scene title *, musical notes.
_NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|[♪♫]")
_SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def clean(text: str) -> str:
    text = _NON_SPEECH.sub(" ", text)
    # a cue's text may wrap across lines; they are one utterance
    text = re.sub(r"\s+", " ", text.replace("\n", " "))
    return text.strip()


def parse_srt(path: Path):
    """Yield (start, end, text) per cue."""
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip())
    for block in blocks:
        m = _SRT_TIME.search(block)
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        text = block[m.end():].strip()
        yield start, end, text


def parse_json(path: Path):
    """youtube_transcript_api JSONFormatter output: [{text, start, duration}]."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))  # tolerate a BOM
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of {text, start, duration}")
    for cue in data:
        start = float(cue["start"])
        yield start, start + float(cue.get("duration", 0.0)), cue.get("text", "")


def split_words(start: float, end: float, text: str):
    """Distribute a cue's span across its words by character length. The result
    is a plausible-looking guess, not a measurement -- see the module docstring."""
    words = text.split()
    total = sum(len(w) for w in words) or 1
    span = max(end - start, 1e-3)
    t = start
    for w in words:
        w_end = t + span * len(w) / total
        yield t, w_end, w
        t = w_end


def to_results(cues, granularity: str, confidence: float):
    results = []
    for start, end, raw in cues:
        text = clean(raw)
        if not text:
            continue
        pieces = ([(start, end, text)] if granularity == "cue"
                  else list(split_words(start, end, text)))
        for w_start, w_end, content in pieces:
            if w_end <= w_start:
                continue
            results.append({
                "type": "word",
                "start_time": round(w_start, 3),
                "end_time": round(w_end, 3),
                "alternatives": [{"content": content, "confidence": confidence, "language": "fa"}],
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="dir of <video_id>.srt / <video_id>.json caption files")
    parser.add_argument("--out", default="./sm_data/transcripts")
    parser.add_argument("--granularity", choices=["cue", "word"], default="cue")
    parser.add_argument(
        "--confidence", type=float, default=1.0,
        help="confidence stamped on every entry. At the default 1.0 sm_04's "
        "--min-confidence gate becomes a no-op for these files; lower it only "
        "if you want imported captions ranked below Speechmatics output",
    )
    parser.add_argument("--force", action="store_true",
                        help="overwrite transcripts that already exist")
    parser.add_argument("--only", default=None, help="import just this video id")
    args = parser.parse_args()

    src_dir, out_dir = Path(args.src), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in (".srt", ".json"))
    if args.only:
        files = [p for p in files if p.stem == args.only]
        if not files:
            sys.exit(f"no caption file for {args.only} in {src_dir}")
    if not files:
        sys.exit(f"no .srt or .json caption files in {src_dir}")

    written, skipped = 0, []
    for path in files:
        dest = out_dir / f"{path.stem}.json"
        if dest.exists() and not args.force:
            print(f"{path.stem}: transcript exists, skipping (--force to replace)", file=sys.stderr)
            skipped.append(path.stem)
            continue
        try:
            cues = parse_srt(path) if path.suffix.lower() == ".srt" else parse_json(path)
            results = to_results(cues, args.granularity, args.confidence)
        except Exception as exc:
            print(f"{path.stem}: FAILED to parse: {exc}", file=sys.stderr)
            skipped.append(path.stem)
            continue
        if not results:
            print(f"{path.stem}: no usable cues after cleaning, skipping", file=sys.stderr)
            skipped.append(path.stem)
            continue

        dest.write_text(json.dumps({
            "format": "2.1",
            "metadata": {"imported_from": path.name, "granularity": args.granularity},
            "results": results,
        }, ensure_ascii=False), encoding="utf-8")
        span = results[-1]["end_time"] - results[0]["start_time"]
        print(f"{path.stem}: {len(results)} entries over {span / 60:.1f} min -> {dest}",
              file=sys.stderr)
        written += 1

    print(f"\nimported {written}/{len(files)} -> {out_dir}")
    if skipped:
        print("skipped: " + ", ".join(skipped), file=sys.stderr)


if __name__ == "__main__":
    main()

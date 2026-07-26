#!/usr/bin/env python3
"""
Step 3 -- transcribe each downloaded file with the Speechmatics batch API,
saving the word-level json-v2 output that sm_04 slices into training clips.

This is the ONLY step that needs the proxy (the server can't reach
speechmatics.com directly). Configure it with env vars so no secret is ever
written into a file:

    export SM_API_KEY=...                       # Speechmatics API key
    export SM_PROXY=http://user:pass@host:3128  # omit to go direct

The .m4a is uploaded as-is rather than converted to wav first: it's ~10x
smaller over the proxy, Speechmatics decodes it server-side, and the returned
timings are on the same timeline as the local file either way.

Usage:
    python sm_03_transcribe.py --audio ./sm_data/audio --out ./sm_data/transcripts
    python sm_03_transcribe.py --only IZNwjfiqeoM      # stress test one file
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API_BASE = "https://asr.api.speechmatics.com/v2"


def build_session(proxy):
    s = requests.Session()
    s.trust_env = False  # never pick up an ambient proxy by accident
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    return s


def submit(session, api_key, audio_path: Path, language: str, operating_point: str) -> str:
    config = {
        "type": "transcription",
        # word-level timings (what sm_04 segments on) come back by default in
        # json-v2; diarization is left off because each clip is a single
        # utterance. Do NOT add enable_partials here -- it is a realtime-only
        # option and the batch API rejects the whole job with HTTP 400.
        "transcription_config": {
            "language": language,
            "operating_point": operating_point,
        },
    }
    with open(audio_path, "rb") as fh:
        r = session.post(
            f"{API_BASE}/jobs",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"data_file": (audio_path.name, fh, "audio/m4a")},
            data={"config": json.dumps(config)},
            timeout=600,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"submit failed HTTP {r.status_code}: {r.text[:400]}")
    return r.json()["id"]


def wait(session, api_key, job_id: str, poll: int, timeout: int) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    started = time.time()
    while True:
        r = session.get(f"{API_BASE}/jobs/{job_id}", headers=headers, timeout=120)
        r.raise_for_status()
        status = r.json()["job"]["status"]
        if status == "done":
            return status
        if status in ("rejected", "expired", "deleted"):
            raise RuntimeError(f"job {job_id} ended as {status}: {r.text[:400]}")
        if time.time() - started > timeout:
            raise TimeoutError(f"job {job_id} still {status} after {timeout}s")
        print(f"    {status}... ({int(time.time() - started)}s)", file=sys.stderr)
        time.sleep(poll)


def fetch_transcript(session, api_key, job_id: str) -> dict:
    r = session.get(
        f"{API_BASE}/jobs/{job_id}/transcript",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"format": "json-v2"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="./sm_data/audio")
    parser.add_argument("--out", default="./sm_data/transcripts")
    parser.add_argument("--language", default="fa")
    parser.add_argument("--operating-point", default="standard", choices=["standard", "enhanced"])
    parser.add_argument("--only", default=None, help="transcribe just this video id (stress test)")
    parser.add_argument("--poll", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    api_key = os.environ.get("SM_API_KEY")
    if not api_key:
        sys.exit("SM_API_KEY is not set (export it; do not hardcode it in a file)")
    proxy = os.environ.get("SM_PROXY")

    session = build_session(proxy)
    print(f"proxy: {'yes' if proxy else 'DIRECT'}", file=sys.stderr)

    # fail fast on auth/proxy problems before uploading hundreds of MB
    probe = session.get(f"{API_BASE}/jobs", headers={"Authorization": f"Bearer {api_key}"},
                        params={"limit": 1}, timeout=120)
    if probe.status_code == 407:
        sys.exit("proxy rejected the credentials (HTTP 407) -- check SM_PROXY user/pass "
                 "and that this machine's IP is allowlisted with the proxy provider")
    if probe.status_code == 401:
        sys.exit("Speechmatics rejected the API key (HTTP 401) -- check SM_API_KEY")
    probe.raise_for_status()
    print("auth + connectivity OK", file=sys.stderr)

    audio_dir, out_dir = Path(args.audio), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(audio_dir.glob("*.m4a"))
    if args.only:
        files = [f for f in files if f.stem == args.only]
        if not files:
            sys.exit(f"no audio file for {args.only} in {audio_dir}")

    failed = []
    for i, path in enumerate(files, 1):
        dest = out_dir / f"{path.stem}.json"
        if dest.exists():
            print(f"[{i}/{len(files)}] {path.stem}: already transcribed, skipping", file=sys.stderr)
            continue
        # a transcript under val/ is a hand-checked label sm_04 holds out; billing
        # Speechmatics to overwrite it would both cost money and leave the id with
        # a transcript in two dirs, which sm_04 rejects
        if (out_dir / "val" / f"{path.stem}.json").exists():
            print(f"[{i}/{len(files)}] {path.stem}: has a validation transcript, skipping",
                  file=sys.stderr)
            continue
        size_mb = path.stat().st_size / 1e6
        print(f"[{i}/{len(files)}] {path.stem}: uploading {size_mb:.1f}MB...", file=sys.stderr)
        t0 = time.time()
        try:
            job_id = submit(session, api_key, path, args.language, args.operating_point)
            print(f"    job {job_id}", file=sys.stderr)
            wait(session, api_key, job_id, args.poll, args.timeout)
            data = fetch_transcript(session, api_key, job_id)
            data["_job_id"] = job_id
            dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            words = sum(1 for r in data.get("results", []) if r.get("type") == "word")
            print(f"    done in {time.time() - t0:.0f}s, {words} words -> {dest}", file=sys.stderr)
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            failed.append(path.stem)

    print(f"\ntranscribed {len(files) - len(failed)}/{len(files)} -> {out_dir}")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

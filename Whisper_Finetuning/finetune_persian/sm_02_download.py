#!/usr/bin/env python3
"""
Step 2 -- download the direct .m4a links from the manifest.

These hosts are domestic (imgurl.ir / uupload.ir / uploadkon.ir) and are
reachable from the server without a proxy, so this step deliberately does NOT
use one -- only the Speechmatics step (sm_03) needs to tunnel out. Requests
here explicitly disable environment proxies so a global HTTPS_PROXY set for
sm_03 can't accidentally route these large downloads through it.

Resumes partial files with a Range request, so an interrupted run can be
re-run without starting over.

Usage:
    python sm_02_download.py --links ./sm_data/links.jsonl --out ./sm_data/audio
"""
import argparse
import json
import sys
from pathlib import Path

import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, dest: Path, timeout: int = 60) -> bool:
    session = requests.Session()
    session.trust_env = False  # ignore HTTP(S)_PROXY -- these links are domestic
    headers = {"User-Agent": UA}

    existing = dest.stat().st_size if dest.exists() else 0
    if existing:
        # verify the server knows the full length before trusting a resume
        head = session.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        total = int(head.headers.get("Content-Length", 0))
        if total and existing >= total:
            print(f"  already complete ({human(existing)})", file=sys.stderr)
            return True
        headers["Range"] = f"bytes={existing}-"
        print(f"  resuming at {human(existing)}", file=sys.stderr)

    with session.get(url, headers=headers, stream=True, timeout=timeout) as r:
        if r.status_code not in (200, 206):
            print(f"  HTTP {r.status_code}", file=sys.stderr)
            return False
        mode = "ab" if r.status_code == 206 else "wb"
        got = existing if r.status_code == 206 else 0
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                print(f"\r  {human(got)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", default="./sm_data/links.jsonl")
    parser.add_argument("--out", default="./sm_data/audio")
    parser.add_argument("--only", default=None, help="download just this video id (stress test)")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # utf-8-sig, in case the manifest was round-tripped through a Windows editor
    entries = [json.loads(l) for l in open(args.links, encoding="utf-8-sig")]
    if args.only:
        entries = [e for e in entries if e["video_id"] == args.only]
        if not entries:
            sys.exit(f"no entry with video_id {args.only}")

    failed = []
    for i, e in enumerate(entries, 1):
        dest = out_dir / f"{e['video_id']}.m4a"
        print(f"[{i}/{len(entries)}] {e['video_id']}", file=sys.stderr)
        try:
            if not download(e["audio_url"], dest, args.timeout):
                failed.append(e["video_id"])
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            failed.append(e["video_id"])

    print(f"\ndownloaded {len(entries) - len(failed)}/{len(entries)} -> {out_dir}")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 1 -- build links.jsonl (the manifest every later step reads) from a plain
text file of direct audio URLs, one per line.

    # comments and blank lines are ignored
    https://host/path/xxxx+IZNwjfiqeoM.m4a
    https://host/path/yyyy.m4a<TAB>my_custom_id

Each URL gets an id used to name its audio file, transcript, and clips. The id
is taken from the trailing token of the filename, which is how these uploaders
embed the source video id; append a TAB and your own id to override it. Ids are
printed so you can eyeball them before downloading hundreds of MB.

A .docx is also accepted (the original input format), in which case the
YouTube/audio/transcript triples are parsed out of the document body.

Usage:
    python sm_01_extract_links.py ./sm_data/urls.txt --out ./sm_data/links.jsonl
"""
import argparse
import html
import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

# YouTube ids are 11 chars of [A-Za-z0-9_-]; these uploaders append the id to
# the stored filename, so the trailing token is the most reliable handle
_ID_LEN = 11


def derive_id(url: str) -> str:
    stem = Path(unquote(urlparse(url).path)).stem
    if "+" in stem:  # uupload.ir style: <random>+<video id>
        stem = stem.rsplit("+", 1)[-1]
    # drop a trailing "_1" duplicate marker, but only when the stem is longer
    # than an id -- ids themselves may legitimately end in _<digits>
    # (e.g. knq6xB7F_88), and stripping those would corrupt them
    if len(stem) > _ID_LEN:
        stem = re.sub(r"_\d+$", "", stem)
    tail = stem[-_ID_LEN:]
    return tail if re.fullmatch(r"[A-Za-z0-9_\-]+", tail) else re.sub(r"[^A-Za-z0-9_\-]", "_", stem)


def read_urls(path: Path):
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"\t+", line)
        url = parts[0].strip()
        explicit = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        entries.append({
            "video_id": explicit or derive_id(url),
            "audio_url": url,
            "youtube_url": None,
            "poor_transcript": "",
        })
    return entries


def read_docx(path: Path):
    """Original input format: (youtube url, direct url, low-quality transcript)
    triples. The transcripts are kept only for comparison, never used as labels."""
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    entries, cur = [], None
    for raw in html.unescape(xml).split("\n"):
        line = re.sub(r'^HYPERLINK\s+"[^"]*"\s*', "", raw.strip()).strip()
        if not line:
            continue
        audio_m = re.match(r"^(https?://\S+?\.m4a)(.*)", line)
        yt_m = re.match(r"^(https://www\.youtube\.com/watch\?v=[\w\-]+)", line)
        if yt_m and not audio_m:
            if cur:
                entries.append(cur)
            cur = {"youtube_url": yt_m.group(1),
                   "video_id": re.search(r"v=([\w\-]+)", yt_m.group(1)).group(1),
                   "audio_url": None, "poor_transcript": ""}
        elif audio_m and cur is not None:
            cur["audio_url"] = audio_m.group(1)
            if audio_m.group(2).strip():  # first entry glues url+transcript
                cur["poor_transcript"] = audio_m.group(2).strip()
        elif cur is not None and cur.get("audio_url"):
            cur["poor_transcript"] = (cur["poor_transcript"] + " " + line).strip()
    if cur:
        entries.append(cur)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="text file of URLs (one per line), or the original .docx")
    parser.add_argument("--out", default="./sm_data/links.jsonl")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="keep entries reusing an audio URL already seen (in the source "
                        ".docx these are copy-paste errors: the audio and transcript belong "
                        "to different videos)")
    args = parser.parse_args()

    src = Path(args.source)
    entries = read_docx(src) if src.suffix.lower() == ".docx" else read_urls(src)

    seen, kept, dropped = {}, [], []
    for e in entries:
        if not e["audio_url"]:
            dropped.append((e["video_id"], "no audio url"))
            continue
        if e["audio_url"] in seen and not args.keep_duplicates:
            dropped.append((e["video_id"], f"duplicate url of {seen[e['audio_url']]}"))
            continue
        seen.setdefault(e["audio_url"], e["video_id"])
        kept.append(e)

    ids = [e["video_id"] for e in kept]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        sys.exit(f"derived ids collide: {sorted(dupes)} -- disambiguate with a TAB-separated "
                 f"explicit id in {src}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"{'id':<16} url", file=sys.stderr)
    for e in kept:
        print(f"{e['video_id']:<16} {e['audio_url']}", file=sys.stderr)
    for vid, why in dropped:
        print(f"dropped {vid}: {why}", file=sys.stderr)
    print(f"\nwrote {len(kept)} entries -> {out_path}")


if __name__ == "__main__":
    main()

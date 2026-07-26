import sys
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter, SRTFormatter, JSONFormatter

video_id = sys.argv[1] if len(sys.argv) > 1 else "RWeO2rRTcXE"
languages = sys.argv[2].split(",") if len(sys.argv) > 2 else None

ytt_api = YouTubeTranscriptApi()
kwargs = {"languages": languages} if languages else {}
transcript = ytt_api.fetch(video_id, **kwargs)

for fmt_name, formatter, ext in [
    ("text", TextFormatter(), "txt"),
    ("srt", SRTFormatter(), "srt"),
    ("json", JSONFormatter(), "json"),
]:
    out_path = f"{video_id}.{ext}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatter.format_transcript(transcript))
    print(f"wrote {fmt_name} -> {out_path}")

"""Download videos and extract public metrics from any platform using yt-dlp."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from .config import detect_platform


def _ytdlp_cmd():
    """Find yt-dlp command - binary or python module."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def get_metadata(url):
    """Get video metadata (views, likes, duration, etc.) without downloading."""
    cmd = _ytdlp_cmd() + ["--dump-json", "--no-download", url]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {result.stderr[:200]}")
    return json.loads(result.stdout)


def extract_public_metrics(url):
    """Extract public metrics from a single video URL."""
    raw = get_metadata(url)
    platform = detect_platform(url)

    metrics = {
        "url": url,
        "platform": platform,
        "id": raw.get("id", ""),
        "title": raw.get("title", ""),
        "uploader": raw.get("uploader", ""),
        "upload_date": raw.get("upload_date", ""),
        "duration": raw.get("duration", 0),
        "view_count": raw.get("view_count", 0),
        "like_count": raw.get("like_count", 0),
        "comment_count": raw.get("comment_count", 0),
        "repost_count": raw.get("repost_count", 0),
        "description": raw.get("description", ""),
        "tags": raw.get("tags", []),
        "thumbnail": raw.get("thumbnail", ""),
    }
    return metrics


def download_video(url, output_dir, quality="worst"):
    """Download a video to output_dir. Returns path to downloaded file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get metadata first for filename
    meta = get_metadata(url)
    video_id = meta.get("id", "video")
    safe_title = "".join(c if c.isalnum() or c in ".-_ " else "" for c in meta.get("title", video_id))[:60]
    filename = f"{safe_title}_{video_id}"

    output_template = str(output_dir / f"{filename}.%(ext)s")
    cmd = _ytdlp_cmd() + [
        "-f", f"{quality}[ext=mp4]/{quality}",
        "--max-filesize", "500M",
        "-o", output_template,
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr[:200]}")

    # Find the downloaded file
    for f in output_dir.iterdir():
        if f.stem.startswith(filename[:20]):
            return str(f), meta
    raise FileNotFoundError(f"Downloaded file not found for {filename}")


def list_channel_videos(channel_url, max_videos=50):
    """List video URLs from a channel/profile."""
    cmd = _ytdlp_cmd() + [
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", str(max_videos),
        channel_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp channel listing failed: {result.stderr[:200]}")

    videos = []
    for line in result.stdout.strip().split("\n"):
        if line:
            data = json.loads(line)
            vid_url = data.get("url") or data.get("webpage_url", "")
            # Reconstruct full URL if needed
            if vid_url and not vid_url.startswith("http"):
                platform = detect_platform(channel_url)
                if platform == "youtube":
                    vid_url = f"https://www.youtube.com/watch?v={vid_url}"
            videos.append({
                "url": vid_url,
                "id": data.get("id", ""),
                "title": data.get("title", ""),
                "duration": data.get("duration", 0),
            })
    return videos

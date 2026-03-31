"""Metrics handling: public extraction + CSV import for private analytics."""
import csv
import json
from pathlib import Path


def parse_youtube_studio_csv(csv_path):
    """Parse YouTube Studio exported CSV (Content tab → Advanced mode → Export)."""
    videos = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append({
                "title": row.get("Video title", row.get("Content", "")),
                "url": row.get("Video URL", ""),
                "views": _int(row.get("Views", 0)),
                "watch_time_hours": _float(row.get("Watch time (hours)", 0)),
                "avg_view_duration": row.get("Average view duration", ""),
                "impressions": _int(row.get("Impressions", 0)),
                "ctr": _float(row.get("Impressions click-through rate (%)", 0)),
                "likes": _int(row.get("Likes", 0)),
                "comments": _int(row.get("Comments", 0)),
                "shares": _int(row.get("Shares", 0)),
                "subscribers_gained": _int(row.get("Subscribers gained", 0)),
                "upload_date": row.get("Video publish time", ""),
            })
    return videos


def parse_tiktok_csv(csv_path):
    """Parse TikTok Analytics exported CSV."""
    videos = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append({
                "title": row.get("Video Description", "")[:80],
                "url": row.get("Video Link", ""),
                "views": _int(row.get("Video Views", 0)),
                "likes": _int(row.get("Likes", 0)),
                "comments": _int(row.get("Comments", 0)),
                "shares": _int(row.get("Shares", 0)),
                "avg_watch_time": _float(row.get("Average Watch Time", 0)),
                "completion_rate": _float(row.get("Watched Full Video (%)", 0)),
                "reach": _int(row.get("Reached Audience", 0)),
                "upload_date": row.get("Date", ""),
            })
    return videos


def parse_instagram_csv(csv_path):
    """Parse Instagram Insights exported CSV (Meta Business Suite)."""
    videos = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append({
                "title": row.get("Description", "")[:80],
                "url": row.get("Permalink", ""),
                "views": _int(row.get("Plays", row.get("Impressions", 0))),
                "likes": _int(row.get("Likes", 0)),
                "comments": _int(row.get("Comments", 0)),
                "shares": _int(row.get("Shares", 0)),
                "saves": _int(row.get("Saves", 0)),
                "reach": _int(row.get("Reach", 0)),
                "upload_date": row.get("Publish time", ""),
            })
    return videos


def parse_generic_csv(csv_path):
    """Parse a generic CSV with at minimum: url, views columns."""
    videos = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append({k.strip().lower(): v for k, v in row.items()})
    return videos


CSV_PARSERS = {
    "youtube": parse_youtube_studio_csv,
    "tiktok": parse_tiktok_csv,
    "instagram": parse_instagram_csv,
    "generic": parse_generic_csv,
}


def load_metrics(csv_path, platform="generic"):
    """Load metrics from a CSV file. Auto-detects platform format if possible."""
    parser = CSV_PARSERS.get(platform, parse_generic_csv)
    return parser(csv_path)


def rank_videos(metrics_list, sort_by="views", top_n=None):
    """Rank videos by a metric. Returns sorted list."""
    key = sort_by.lower()
    sorted_list = sorted(metrics_list, key=lambda x: _int(x.get(key, 0)), reverse=True)
    if top_n:
        return sorted_list[:top_n]
    return sorted_list


def _int(val):
    try:
        return int(str(val).replace(",", "").replace(" ", ""))
    except (ValueError, TypeError):
        return 0


def _float(val):
    try:
        return float(str(val).replace(",", "").replace("%", "").replace(" ", ""))
    except (ValueError, TypeError):
        return 0.0

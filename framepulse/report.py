"""Report generation: cross-reference video analysis with metrics."""
import json
from .analyzer import fmt_time


def cross_reference(analyses, metrics_list):
    """Cross-reference video analyses with metrics to find patterns.

    Args:
        analyses: list of dicts from analyzer.analyze_video()
        metrics_list: list of dicts with at least {url, views, likes, ...}

    Returns:
        dict with ranked videos, patterns found, and recommendations
    """
    # Match analyses with metrics by URL or title
    combined = []
    for analysis in analyses:
        matched_metrics = None
        for m in metrics_list:
            if _urls_match(analysis.get("url", ""), m.get("url", "")):
                matched_metrics = m
                break
        combined.append({
            "analysis": analysis,
            "metrics": matched_metrics or {},
        })

    # Sort by views (highest first)
    combined.sort(key=lambda x: int(x["metrics"].get("views", x["metrics"].get("view_count", 0))),
                  reverse=True)

    # Split into top and bottom performers
    n = len(combined)
    if n >= 4:
        top = combined[:n // 4]
        bottom = combined[-(n // 4):]
    else:
        top = combined[:1]
        bottom = combined[-1:]

    return {
        "total_videos": n,
        "ranked": combined,
        "top_performers": top,
        "bottom_performers": bottom,
    }


def format_text_report(report_data, channel_name=""):
    """Format a human-readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"FRAMEPULSE REPORT{f': {channel_name}' if channel_name else ''}")
    lines.append(f"Videos analyzed: {report_data['total_videos']}")
    lines.append("=" * 70)

    lines.append("\n--- TOP PERFORMERS ---")
    for item in report_data["top_performers"]:
        m = item["metrics"]
        a = item["analysis"]
        views = m.get("views", m.get("view_count", "?"))
        likes = m.get("likes", m.get("like_count", "?"))
        title = m.get("title", a.get("video_path", "Unknown"))
        lines.append(f"\n  [{views:,} views | {likes:,} likes] {title}")
        if a.get("synthesis"):
            lines.append(f"  Analysis: {a['synthesis'][:300]}...")

    lines.append("\n--- BOTTOM PERFORMERS ---")
    for item in report_data["bottom_performers"]:
        m = item["metrics"]
        a = item["analysis"]
        views = m.get("views", m.get("view_count", "?"))
        likes = m.get("likes", m.get("like_count", "?"))
        title = m.get("title", a.get("video_path", "Unknown"))
        lines.append(f"\n  [{views:,} views | {likes:,} likes] {title}")
        if a.get("synthesis"):
            lines.append(f"  Analysis: {a['synthesis'][:300]}...")

    lines.append("\n--- ALL VIDEOS (ranked by views) ---")
    for i, item in enumerate(report_data["ranked"], 1):
        m = item["metrics"]
        views = m.get("views", m.get("view_count", "?"))
        title = m.get("title", "Unknown")
        duration = item["analysis"].get("duration", 0)
        lines.append(f"  {i:3d}. [{views:>10,} views] {fmt_time(duration)} - {title}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def format_json_report(report_data):
    """Format report as JSON."""
    return json.dumps(report_data, indent=2, ensure_ascii=False, default=str)


def _urls_match(url1, url2):
    """Fuzzy match two URLs (handles different formats of same video)."""
    if not url1 or not url2:
        return False
    # Extract video IDs for comparison
    for u in [url1, url2]:
        u = u.strip().rstrip("/")
    return url1.strip() == url2.strip()

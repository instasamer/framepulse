#!/usr/bin/env python3
"""
FramePulse CLI - AI-powered video analysis for creators.

Usage:
  python cli.py analyze video.mp4                         # Single video analysis
  python cli.py spy @channel --last 20                    # Spy on competitor (public metrics)
  python cli.py study @channel --csv metrics.csv          # Your channel (private metrics)
"""
import argparse
import json
import sys
import os
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def cmd_analyze(args):
    """Analyze a single video file or URL."""
    from framepulse.analyzer import analyze_video, fmt_time
    from framepulse.config import MODELS
    from framepulse.downloader import download_video, extract_public_metrics

    preset = MODELS[args.preset]
    video_path = args.video

    # If URL, download first
    if video_path.startswith("http"):
        print(f"Downloading {video_path}...")
        metrics = extract_public_metrics(video_path)
        print(f"  {metrics['title']} ({metrics['view_count']:,} views)")
        tmpdir = Path("./framepulse_tmp")
        video_path, _ = download_video(args.video, tmpdir)
        print(f"  Saved to {video_path}")

    result = analyze_video(
        video_path,
        model_id=preset["vlm"],
        whisper_model=preset["whisper"],
        fps=args.fps or preset["fps"],
        batch_size=args.batch_size or preset["batch_size"],
        hint=args.hint or "",
        no_audio=args.no_audio,
        verbose=args.verbose,
    )

    _output_result(result, args)


def cmd_spy(args):
    """Spy mode: analyze competitor's public videos."""
    from framepulse.downloader import list_channel_videos, download_video, extract_public_metrics
    from framepulse.analyzer import analyze_video
    from framepulse.report import cross_reference, format_text_report, format_json_report
    from framepulse.config import MODELS

    preset = MODELS[args.preset]
    channel = args.channel

    # List videos
    print(f"Listing videos from {channel}...")
    videos = list_channel_videos(channel, max_videos=args.last)
    print(f"  Found {len(videos)} videos")

    tmpdir = Path("./framepulse_tmp")
    analyses = []
    all_metrics = []

    for i, vid in enumerate(videos, 1):
        url = vid["url"]
        print(f"\n[{i}/{len(videos)}] {vid.get('title', url)}")

        try:
            # Get public metrics
            metrics = extract_public_metrics(url)
            all_metrics.append(metrics)
            print(f"  {metrics['view_count']:,} views | {metrics['like_count']:,} likes")

            # Download
            video_path, _ = download_video(url, tmpdir, quality="worst")

            # Analyze
            result = analyze_video(
                video_path,
                model_id=preset["vlm"],
                whisper_model=preset["whisper"],
                fps=args.fps or preset["fps"],
                batch_size=args.batch_size or preset["batch_size"],
                no_audio=args.no_audio,
                verbose=args.verbose,
            )
            result["url"] = url
            result["title"] = metrics.get("title", "")
            analyses.append(result)

            # Clean up video to save disk
            os.unlink(video_path)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Generate report
    print(f"\nGenerating report...")
    report = cross_reference(analyses, all_metrics)
    channel_name = videos[0].get("title", channel) if videos else channel

    if args.json:
        output = format_json_report(report)
    else:
        output = format_text_report(report, channel_name=channel)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Saved to {args.output}")

    print(output)


def cmd_study(args):
    """Study mode: analyze your own channel with private metrics CSV."""
    from framepulse.downloader import download_video
    from framepulse.analyzer import analyze_video
    from framepulse.metrics import load_metrics, rank_videos
    from framepulse.report import cross_reference, format_text_report, format_json_report
    from framepulse.config import MODELS

    preset = MODELS[args.preset]

    # Load private metrics
    print(f"Loading metrics from {args.csv}...")
    metrics_list = load_metrics(args.csv, platform=args.platform)
    print(f"  {len(metrics_list)} videos found")

    # Optionally filter to top/bottom N
    if args.top:
        metrics_list = rank_videos(metrics_list, sort_by="views", top_n=args.top)
        print(f"  Analyzing top {args.top} by views")

    tmpdir = Path("./framepulse_tmp")
    analyses = []

    for i, m in enumerate(metrics_list, 1):
        url = m.get("url", "")
        title = m.get("title", url)
        print(f"\n[{i}/{len(metrics_list)}] {title}")

        if not url:
            print("  SKIP: no URL")
            continue

        try:
            video_path, _ = download_video(url, tmpdir, quality="worst")
            result = analyze_video(
                video_path,
                model_id=preset["vlm"],
                whisper_model=preset["whisper"],
                fps=args.fps or preset["fps"],
                batch_size=args.batch_size or preset["batch_size"],
                no_audio=args.no_audio,
                verbose=args.verbose,
            )
            result["url"] = url
            result["title"] = title
            analyses.append(result)
            os.unlink(video_path)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nGenerating report...")
    report = cross_reference(analyses, metrics_list)

    if args.json:
        output = format_json_report(report)
    else:
        output = format_text_report(report, channel_name=args.channel or "")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Saved to {args.output}")

    print(output)


def _output_result(result, args):
    """Output a single video analysis result."""
    from framepulse.analyzer import fmt_time

    if args.json:
        output = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    else:
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"FRAMEPULSE: {Path(result['video_path']).stem}")
        t = result['timings']
        lines.append(f"Duration: {fmt_time(result['duration'])} | Frames: {result['frames_analyzed']}")
        lines.append(f"Time: {t['total']:.0f}s (extract={t['extract']:.0f}s whisper={t['whisper']:.0f}s vlm={t['vlm']:.0f}s)")
        lines.append(f"{'='*60}")

        lines.append("\n--- VISUAL TIMELINE ---")
        for d in result["visual_timeline"]:
            lines.append(f"[{fmt_time(d['start'])}-{fmt_time(d['end'])}] {d['description']}")

        if result["transcript"]:
            lines.append("\n--- TRANSCRIPT ---")
            for s in result["transcript"]:
                lines.append(f"[{fmt_time(s['start'])}] \"{s['text']}\"")

        lines.append("\n--- SYNTHESIS ---")
        lines.append(result["synthesis"])
        lines.append(f"\n{'='*60}")
        output = "\n".join(lines)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Saved to {args.output}")

    print(output)


def main():
    parser = argparse.ArgumentParser(
        description="FramePulse - AI-powered video analysis for creators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py analyze video.mp4 -v
  python cli.py analyze "https://youtube.com/watch?v=..." --preset pod
  python cli.py spy "https://youtube.com/@channel" --last 20 -o report.txt
  python cli.py study --csv youtube_export.csv --platform youtube --top 30
        """
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # Shared args added to each subparser so order doesn't matter
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--preset", default="pod", choices=["local-small", "local-medium", "pod"],
                        help="GPU/model preset (default: pod)")
    shared.add_argument("--fps", type=int, help="Frames per second (overrides preset)")
    shared.add_argument("--batch-size", type=int, help="Frames per VLM batch (overrides preset)")
    shared.add_argument("--no-audio", action="store_true", help="Skip audio transcription")
    shared.add_argument("--json", action="store_true", help="Output as JSON")
    shared.add_argument("-o", "--output", help="Save output to file")
    shared.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # analyze
    p_analyze = sub.add_parser("analyze", parents=[shared], help="Analyze a single video (file or URL)")
    p_analyze.add_argument("video", help="Video file path or URL")
    p_analyze.add_argument("--hint", help="Description hint for better analysis")

    # spy
    p_spy = sub.add_parser("spy", parents=[shared], help="Spy on a channel (public metrics only)")
    p_spy.add_argument("channel", help="Channel/profile URL")
    p_spy.add_argument("--last", type=int, default=20, help="Number of recent videos (default: 20)")

    # study
    p_study = sub.add_parser("study", parents=[shared], help="Study your channel with private metrics")
    p_study.add_argument("--csv", required=True, help="Path to analytics CSV export")
    p_study.add_argument("--channel", help="Channel name for report header")
    p_study.add_argument("--platform", default="generic",
                         choices=["youtube", "tiktok", "instagram", "generic"],
                         help="CSV format (default: generic)")
    p_study.add_argument("--top", type=int, help="Only analyze top N videos by views")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "spy":
        cmd_spy(args)
    elif args.command == "study":
        cmd_study(args)


if __name__ == "__main__":
    main()

# FramePulse

**AI-powered video analysis that tells creators why their videos work (or don't).**

FramePulse crosses frame-by-frame AI analysis with performance metrics to find patterns in what makes content go viral. Works with YouTube, TikTok, Instagram, and X. Stop guessing, start knowing.

## 3 modes

```bash
# Analyze a single video (local file or any URL)
python cli.py analyze video.mp4
python cli.py analyze "https://tiktok.com/@user/video/123"

# Spy on a competitor (public metrics + AI analysis)
python cli.py spy "https://youtube.com/@competitor" --last 30

# Study your own channel (private analytics CSV + AI analysis)
python cli.py study --csv my_analytics.csv --platform youtube --top 50
```

## What it does

1. **Downloads videos** from any platform (YouTube, TikTok, Instagram, X) via yt-dlp
2. **Extracts frames** at 4fps for dense visual analysis
3. **Transcribes audio** with OpenAI Whisper
4. **Analyzes every frame** with Qwen3.5 vision AI in VRAM-safe batches
5. **Pulls metrics** - public (views, likes) or private (retention, CTR from your exports)
6. **Finds patterns** - correlates visual/audio elements with performance

## Example insights

- "Your videos with intros under 5 seconds have 3x higher retention"
- "Split-screen segments correlate with 40% higher CTR"
- "Videos where you show the result first get 2.5x more views"

## GPU presets

| Preset | GPU | VRAM | VLM | Whisper |
|--------|-----|------|-----|---------|
| `local-small` | RTX 3070 | 8GB | Qwen3.5-0.8B | turbo |
| `local-medium` | RTX 4080 | 16GB | Qwen3.5-4B | turbo |
| `pod` (default) | RTX 5090 / A100 | 32GB+ | Qwen3.5-9B | large-v3 |

## Run on RunPod (recommended)

See [runpod/README.md](runpod/README.md) for setup instructions. Cost: ~$2-4 to analyze 50 videos.

## Local install

```bash
git clone https://github.com/instasamer/framepulse.git
cd framepulse
pip install -r requirements.txt
python cli.py analyze video.mp4 --preset local-small -v
```

## Project structure

```
framepulse/
  __init__.py       # Package init
  analyzer.py       # Core: frame extraction + Whisper + VLM analysis
  config.py         # Model presets, constants
  downloader.py     # yt-dlp wrapper, multi-platform download + metadata
  metrics.py        # CSV parsers for YouTube/TikTok/Instagram analytics
  report.py         # Cross-reference analysis x metrics, generate reports
cli.py              # CLI entry point (analyze/spy/study)
runpod/
  bootstrap.sh      # One-command pod setup
  README.md         # RunPod instructions
```

## Status

Early development. Core video analysis engine working. Cross-referencing with metrics in progress.

## License

MIT

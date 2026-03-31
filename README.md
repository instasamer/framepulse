# FramePulse

**AI-powered video analysis that tells creators why their videos work (or don't).**

FramePulse crosses frame-by-frame AI analysis with YouTube performance metrics to find patterns in what makes content go viral. Stop guessing, start knowing.

## What it does

1. **Analyzes videos** - Dense frame extraction + audio transcription + vision AI describes every second of your content
2. **Pulls metrics** - Views, retention, CTR, likes, comments from YouTube API
3. **Finds patterns** - Correlates visual/audio elements with performance to answer: *"Why did this video get 2M views and this other one 50K?"*

## Example insights

- "Your videos with intros under 5 seconds have 3x higher retention"
- "Split-screen segments correlate with 40% higher CTR"
- "Videos where you show the result first get 2.5x more views"

## Tech stack

- **Vision AI**: Qwen3.5 (0.8B for local, 9B+ for cloud/pod)
- **Audio**: OpenAI Whisper (turbo/large-v3)
- **Frame extraction**: OpenCV + FFmpeg
- **Metrics**: YouTube Data API v3
- **GPU**: Runs on RTX 3070 (8GB) locally, or cloud GPU pods for best results

## Quick start

```bash
# Analyze a single video
python framepulse.py video.mp4

# Analyze with YouTube metrics
python framepulse.py --channel UC_x5XG1OV2P6uZZ5FSM9Ttw --last 50

# Use a bigger model on a GPU pod
python framepulse.py video.mp4 --model Qwen/Qwen3.5-9B
```

## Status

Early development. Built at the intersection of AI video understanding and creator analytics.

## License

MIT

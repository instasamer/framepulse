# Running FramePulse on RunPod

## Quick start

1. Create a pod on [RunPod](https://runpod.io) with:
   - GPU: RTX 5090 / A100 / H100 (32GB+ VRAM recommended)
   - Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   - Disk: 50GB minimum

2. SSH into the pod and run:
```bash
git clone https://github.com/instasamer/framepulse.git
cd framepulse
bash runpod/bootstrap.sh
```

3. Run your analysis:
```bash
# Spy on a competitor
python cli.py spy "https://youtube.com/@MrBeast" --last 20 -o report.txt -v

# Analyze a single video
python cli.py analyze "https://tiktok.com/@user/video/123" -v

# Study your own channel with exported metrics
python cli.py study --csv my_youtube_analytics.csv --platform youtube --top 30
```

## Estimated costs

| Videos | Time | Cost (5090 @ $0.69/hr) |
|--------|------|------------------------|
| 1 | ~5 min | $0.06 |
| 20 | ~1.5 hr | $1.00 |
| 50 | ~3.5 hr | $2.40 |

## Tips

- Use `--preset pod` (default) for best quality on 32GB+ GPUs
- Use `--no-audio` to skip transcription and save time if you only need visual analysis
- Videos are downloaded and deleted after analysis to save disk space
- Stop the pod when done to avoid charges

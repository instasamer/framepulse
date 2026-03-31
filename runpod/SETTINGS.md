# RunPod Configuration Guide

## Pod Settings

### Option A: RTX 5090 (recommended)
- **GPU Type**: RTX 5090 (32GB VRAM)
- **Template**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container Disk**: 20 GB (code + temp files)
- **Volume Disk**: 50 GB (models + downloaded videos)
- **vCPU**: 4+
- **RAM**: 16 GB+
- **Cost**: ~$0.69/hr

### Option B: A100 (faster, pricier)
- **GPU Type**: A100 80GB
- **Template**: same as above
- **Container Disk**: 20 GB
- **Volume Disk**: 50 GB
- **Cost**: ~$1.64/hr

### Option C: H100 (overkill but fast)
- **GPU Type**: H100 80GB
- **Template**: same as above
- **Container Disk**: 20 GB
- **Volume Disk**: 50 GB
- **Cost**: ~$3.89/hr

## Step by step

### 1. Create pod

Go to https://runpod.io/console/pods → Deploy

- Click "GPU Cloud"
- Select GPU (RTX 5090 recommended)
- Template: search "pytorch" → select `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Container Disk: 20 GB
- Volume Disk: 50 GB (or 0 if you don't want persistent storage)
- Click "Deploy On-Demand"

### 2. Connect

Once the pod is running:
- Click "Connect" → "Start Web Terminal" or "SSH"
- For SSH: `ssh root@{pod-ip} -p {port} -i ~/.ssh/id_ed25519`

### 3. Setup FramePulse

```bash
cd /workspace
git clone https://github.com/instasamer/framepulse.git
cd framepulse
bash runpod/bootstrap.sh
```

First run takes ~3-5 min (model downloads). Subsequent runs on same volume: instant.

### 4. Run analysis

```bash
# Spy on a channel
python cli.py spy "https://youtube.com/@MrBeast" --last 20 -o report.txt -v

# Single video
python cli.py analyze "https://tiktok.com/@user/video/123" -v -o analysis.txt

# Your channel with CSV
python cli.py study --csv /workspace/my_analytics.csv --platform youtube -o report.txt
```

### 5. Get results

```bash
# Download results to your PC
# From your local terminal:
scp -P {port} root@{pod-ip}:/workspace/framepulse/report.txt ./
```

Or just cat the file in the web terminal and copy-paste.

### 6. Stop pod

**IMPORTANT**: Stop or terminate the pod when done to avoid charges.
- "Stop" keeps the volume (fast restart later, pays $0.07/GB/month for volume)
- "Terminate" deletes everything (no ongoing cost, but re-download models next time)

## Cost estimates

| Task | Videos | GPU | Time | Cost |
|------|--------|-----|------|------|
| Quick spy | 10 | 5090 | ~45 min | $0.50 |
| Full channel | 50 | 5090 | ~3.5 hr | $2.40 |
| Deep study | 100 | 5090 | ~7 hr | $4.80 |
| Quick spy | 10 | A100 | ~30 min | $0.80 |
| Full channel | 50 | A100 | ~2.5 hr | $4.10 |

## Troubleshooting

### "No GPU available in this region"
Try a different region. RunPod has datacenters in US, EU, and APAC.
RTX 5090s are newest so availability varies. A100s are more widely available.

### "CUDA out of memory"
The pod preset uses batch_size=96 for 32GB VRAM. If using a 24GB GPU:
```bash
python cli.py spy ... --batch-size 64
```

### Models downloading slowly
HuggingFace Hub can be slow sometimes. Set a token for faster downloads:
```bash
export HF_TOKEN=your_token_here
```

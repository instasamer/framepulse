#!/bin/bash
# FramePulse RunPod Bootstrap
# Run this once when the pod starts. Downloads models + deps in parallel.
set -e

echo "=== FramePulse RunPod Bootstrap ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Install system deps
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1 &
APT_PID=$!

# Install Python deps
pip install -q torch transformers accelerate opencv-python openai-whisper \
    Pillow requests yt-dlp huggingface_hub 2>&1 | tail -3 &
PIP_PID=$!

# Wait for pip before downloading models (need huggingface_hub)
wait $PIP_PID
echo "Python deps installed."

# Download models in parallel
echo "Downloading models..."
python3 -c "
from huggingface_hub import snapshot_download
import concurrent.futures

models = [
    'Qwen/Qwen3.5-9B',
    # Whisper downloads on first use via openai-whisper
]

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = {pool.submit(snapshot_download, m): m for m in models}
    for f in concurrent.futures.as_completed(futures):
        print(f'  Downloaded {futures[f]}')
" &
MODEL_PID=$!

# Pre-download Whisper large-v3 in parallel
python3 -c "
import whisper
print('  Downloading Whisper large-v3...')
whisper._download(whisper._MODELS['large-v3'], '/root/.cache/whisper', False)
print('  Whisper large-v3 ready.')
" &
WHISPER_PID=$!

# Wait for everything
wait $APT_PID 2>/dev/null
echo "System deps installed."
wait $MODEL_PID
echo "VLM model ready."
wait $WHISPER_PID
echo "Whisper model ready."

echo ""
echo "=== Bootstrap complete ==="
echo "Run: python cli.py analyze/spy/study ..."

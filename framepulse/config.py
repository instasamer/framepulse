"""Configuration and model presets."""

# Model presets by GPU tier
MODELS = {
    "local-small": {
        "vlm": "Qwen/Qwen3.5-0.8B",
        "whisper": "turbo",
        "batch_size": 64,
        "fps": 4,
        "description": "RTX 3070 / 8GB VRAM"
    },
    "local-medium": {
        "vlm": "Qwen/Qwen3.5-4B",
        "whisper": "turbo",
        "batch_size": 48,
        "fps": 4,
        "description": "RTX 4080 / 16GB VRAM"
    },
    "pod": {
        "vlm": "Qwen/Qwen3.5-9B",
        "whisper": "large-v3",
        "batch_size": 96,
        "fps": 4,
        "description": "RTX 5090 / 32GB VRAM (RunPod)"
    },
}

DEFAULT_PRESET = "pod"
FRAME_SIZE = (384, 384)
MAX_NEW_TOKENS_BATCH = 250
MAX_NEW_TOKENS_SYNTHESIS = 600

# Supported platforms
PLATFORMS = {
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "tiktok.com": "tiktok",
    "instagram.com": "instagram",
    "x.com": "x",
    "twitter.com": "x",
}

def detect_platform(url):
    """Detect platform from URL."""
    for domain, platform in PLATFORMS.items():
        if domain in url:
            return platform
    return "unknown"

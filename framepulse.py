"""
FramePulse - AI-powered video analysis
Tells creators why their videos work (or don't).
https://github.com/instasamer/framepulse
"""
import argparse, sys, os, tempfile, subprocess, time, json, gc
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import cv2
import numpy as np
from PIL import Image

# ─── Constants ───────────────────────────────────────────────
DEFAULT_FPS = 4
BATCH_SIZE = 64
MAX_NEW_TOKENS = 512
WHISPER_MODEL = "turbo"
VLM_MODEL = "Qwen/Qwen3.5-0.8B"
FRAME_SIZE = (384, 384)


def fmt_time(seconds):
    """Format seconds as M:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# ─── Stage 1: Extract ───────────────────────────────────────
def extract_audio(video_path, verbose=False):
    """Extract audio from video as 16kHz mono WAV."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        tmp.name
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if verbose:
            print(f"  [!] No audio track found or ffmpeg error")
        os.unlink(tmp.name)
        return None
    if os.path.getsize(tmp.name) < 1000:
        os.unlink(tmp.name)
        return None
    return tmp.name


def extract_frames(video_path, fps=DEFAULT_FPS, verbose=False):
    """Extract frames at target FPS, return list of (timestamp, PIL Image)."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / video_fps if video_fps > 0 else 0
    interval = max(1, int(video_fps / fps))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            ts = idx / video_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, FRAME_SIZE)
            frames.append((ts, Image.fromarray(resized)))
        idx += 1
    cap.release()

    if verbose:
        print(f"  Extracted {len(frames)} frames from {fmt_time(duration)} video ({fps} fps)")
    return frames, duration


# ─── Stage 2: Transcribe ────────────────────────────────────
def transcribe_audio(wav_path, model_size=WHISPER_MODEL, verbose=False):
    """Transcribe audio using Whisper on GPU, then free VRAM."""
    import torch
    import whisper

    if verbose:
        print(f"  Loading Whisper {model_size}...")
    t0 = time.time()
    model = whisper.load_model(model_size, device="cuda")
    result = model.transcribe(wav_path, language=None, verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    # Free VRAM
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        lang = result.get("language", "?")
        print(f"  Transcribed {len(segments)} segments (lang={lang}) in {time.time()-t0:.1f}s")
    return segments


# ─── Stage 3: Visual Analysis ───────────────────────────────
def load_vlm(verbose=False):
    """Load Qwen3.5-0.8B VLM to GPU."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    if verbose:
        print(f"  Loading {VLM_MODEL}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(VLM_MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        VLM_MODEL, dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    if verbose:
        print(f"  VLM loaded in {time.time()-t0:.1f}s")
    return model, processor


def analyze_batch(model, processor, frames_pil, batch_start, batch_end, project_hint=""):
    """Analyze a batch of frames with the VLM."""
    import torch

    content = []
    for img in frames_pil:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": (
        f"These are {len(frames_pil)} sequential frames from a video "
        f"(covering {fmt_time(batch_start)} to {fmt_time(batch_end)}). "
        f"{project_hint}"
        "List what you see: 1) Screen layout and colors 2) Text/labels visible on screen "
        "3) User actions (clicks, typing, navigation) 4) Any animations or transitions. "
        "Do NOT repeat yourself. Be factual and specific. Max 120 words."
    )})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=250,
            repetition_penalty=1.2,
            no_repeat_ngram_size=6,
        )
    response = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Free intermediate tensors
    del inputs, output
    torch.cuda.empty_cache()
    return response


def analyze_frames(frames, model, processor, batch_size=BATCH_SIZE, project_hint="", verbose=False):
    """Analyze all frames in batches."""
    import torch
    descriptions = []
    total_batches = (len(frames) + batch_size - 1) // batch_size

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_num = i // batch_size + 1
        batch_start = batch[0][0]
        batch_end = batch[-1][0]
        batch_pil = [f[1] for f in batch]

        if verbose:
            print(f"  Batch {batch_num}/{total_batches} [{fmt_time(batch_start)}-{fmt_time(batch_end)}] ({len(batch)} frames)")

        try:
            desc = analyze_batch(model, processor, batch_pil, batch_start, batch_end, project_hint)
        except torch.cuda.OutOfMemoryError:
            # Auto-reduce batch size
            half = len(batch_pil) // 2
            if verbose:
                print(f"  [!] OOM - retrying with {half} frames")
            torch.cuda.empty_cache()
            desc = analyze_batch(model, processor, batch_pil[:half], batch_start,
                                 batch[half-1][0], project_hint)

        descriptions.append({
            "start": batch_start,
            "end": batch_end,
            "description": desc
        })

    return descriptions


# ─── Stage 4: Synthesis ─────────────────────────────────────
def synthesize(model, processor, visual_descs, transcript, verbose=False):
    """Combine visual descriptions + transcript into final summary (text-only, minimal VRAM)."""
    import torch

    # Build visual timeline text
    visual_text = ""
    for d in visual_descs:
        visual_text += f"[{fmt_time(d['start'])}-{fmt_time(d['end'])}] {d['description']}\n"

    # Build transcript text
    transcript_text = ""
    for seg in (transcript or []):
        transcript_text += f"[{fmt_time(seg['start'])}] \"{seg['text']}\"\n"

    prompt = f"""Analyze this hackathon demo video. Be concise, no filler.

1. SUMMARY (2-3 sentences): What does this project do? What problem does it solve?
2. BEST TWITTER MOMENTS (2-3 with timestamps): Which 15-30 second clips would stop someone from scrolling on Twitter/X? Give exact timestamps and explain the visual hook.
3. VIRALITY SCORE (1-10): How likely is this to go viral on Crypto Twitter? Why?

VISUAL TIMELINE:
{visual_text}

TRANSCRIPT:
{transcript_text if transcript_text else "(no audio)"}
"""
    # Truncate if too long
    if len(prompt) > 30000:
        prompt = prompt[:30000] + "\n...(truncated)"

    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

    if verbose:
        print("  Generating synthesis...")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )
    response = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    del inputs, output
    torch.cuda.empty_cache()
    return response


# ─── Stage 5: Output ────────────────────────────────────────
def format_output(video_path, duration, frames_count, visual_descs, transcript, synthesis, timings):
    """Format everything into readable output."""
    name = Path(video_path).stem
    batches = len(visual_descs)

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"VIDEOSCAN: {name}")
    lines.append(f"Duration: {fmt_time(duration)} | Frames analyzed: {frames_count} | Batches: {batches}")
    lines.append(f"Timings: extract={timings['extract']:.1f}s whisper={timings['whisper']:.1f}s vlm={timings['vlm']:.1f}s synthesis={timings['synthesis']:.1f}s total={timings['total']:.1f}s")
    lines.append(f"{'='*60}")

    lines.append("\n--- VISUAL TIMELINE ---")
    for d in visual_descs:
        lines.append(f"[{fmt_time(d['start'])}-{fmt_time(d['end'])}] {d['description']}")

    if transcript:
        lines.append("\n--- TRANSCRIPT ---")
        for seg in transcript:
            lines.append(f"[{fmt_time(seg['start'])}] \"{seg['text']}\"")

    lines.append(f"\n--- SYNTHESIS ---")
    lines.append(synthesis)

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


def build_json(video_path, duration, frames_count, visual_descs, transcript, synthesis, timings):
    """Build JSON output."""
    return {
        "video": Path(video_path).name,
        "duration": duration,
        "frames_analyzed": frames_count,
        "timings": timings,
        "visual_timeline": visual_descs,
        "transcript": transcript,
        "synthesis": synthesis,
    }


# ─── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VideoScan - Local AI video analysis")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Frames per second (default: {DEFAULT_FPS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Frames per VLM batch (default: {BATCH_SIZE})")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL, help=f"Whisper model (default: {WHISPER_MODEL})")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio transcription")
    parser.add_argument("--hint", default="", help="Project name/description hint for better analysis")
    parser.add_argument("-o", "--output", help="Save output to file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress details")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: {args.video} not found")
        sys.exit(1)

    total_start = time.time()
    timings = {}

    # Stage 1: Extract
    print(f"\n[1/4] Extracting frames and audio...")
    t0 = time.time()
    frames, duration = extract_frames(args.video, fps=args.fps, verbose=args.verbose)
    wav_path = None if args.no_audio else extract_audio(args.video, verbose=args.verbose)
    timings['extract'] = time.time() - t0
    print(f"  Done: {len(frames)} frames, {fmt_time(duration)} duration")

    # Stage 2: Transcribe
    transcript = None
    if wav_path:
        print(f"\n[2/4] Transcribing audio with Whisper {args.whisper_model}...")
        t0 = time.time()
        transcript = transcribe_audio(wav_path, model_size=args.whisper_model, verbose=args.verbose)
        timings['whisper'] = time.time() - t0
        os.unlink(wav_path)
        print(f"  Done: {len(transcript)} segments in {timings['whisper']:.1f}s")
    else:
        timings['whisper'] = 0
        print(f"\n[2/4] Skipping audio transcription")

    # Stage 3: Visual analysis
    print(f"\n[3/4] Analyzing video with {VLM_MODEL}...")
    t0 = time.time()
    model, processor = load_vlm(verbose=args.verbose)
    project_hint = f"This is a demo for '{args.hint}'. " if args.hint else ""
    visual_descs = analyze_frames(frames, model, processor, batch_size=args.batch_size,
                                   project_hint=project_hint, verbose=args.verbose)
    timings['vlm'] = time.time() - t0
    print(f"  Done: {len(visual_descs)} batches in {timings['vlm']:.1f}s")

    # Stage 4: Synthesis
    print(f"\n[4/4] Generating synthesis...")
    t0 = time.time()
    synthesis = synthesize(model, processor, visual_descs, transcript, verbose=args.verbose)
    timings['synthesis'] = time.time() - t0

    # Free GPU
    import torch
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    timings['total'] = time.time() - total_start
    print(f"  Done in {timings['total']:.1f}s total")

    # Stage 5: Output
    if args.json:
        result = build_json(args.video, duration, len(frames), visual_descs, transcript, synthesis, timings)
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = format_output(args.video, duration, len(frames), visual_descs, transcript, synthesis, timings)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\nSaved to {args.output}")

    print(f"\n{output}")


if __name__ == "__main__":
    main()

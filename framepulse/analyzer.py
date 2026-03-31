"""Core video analysis engine: frame extraction + Whisper + VLM."""
import sys, os, shutil, tempfile, subprocess, time, gc
import cv2
import numpy as np
from PIL import Image
from .config import FRAME_SIZE, MAX_NEW_TOKENS_BATCH, MAX_NEW_TOKENS_SYNTHESIS

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def extract_audio(video_path):
    """Extract audio as 16kHz mono WAV. Returns path or None."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-y", "-i", video_path, "-vn",
           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp.name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or os.path.getsize(tmp.name) < 1000:
        os.unlink(tmp.name)
        return None
    return tmp.name


def extract_frames(video_path, fps=4):
    """Extract frames at target FPS. Returns [(timestamp, PIL Image), ...] and duration."""
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
    return frames, duration


def transcribe(wav_path, model_size="turbo", verbose=False):
    """Transcribe audio with Whisper on GPU. Frees VRAM after."""
    import torch
    import whisper

    if verbose:
        print(f"  Loading Whisper {model_size}...")
    model = whisper.load_model(model_size, device="cuda")
    result = model.transcribe(wav_path, language=None, verbose=False)

    segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result.get("segments", [])]
    language = result.get("language", "unknown")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return segments, language


def load_vlm(model_id, verbose=False):
    """Load VLM to GPU."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    if verbose:
        print(f"  Loading {model_id}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    if verbose:
        print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, processor


def analyze_batch(model, processor, frames_pil, batch_start, batch_end, hint=""):
    """Analyze a single batch of frames."""
    import torch

    content = [{"type": "image", "image": img} for img in frames_pil]
    content.append({"type": "text", "text": (
        f"These are {len(frames_pil)} sequential frames ({fmt_time(batch_start)}-{fmt_time(batch_end)}). "
        f"{hint}"
        "List what you see: 1) Screen layout and colors 2) Text/labels visible "
        "3) User actions (clicks, typing, navigation) 4) Animations or transitions. "
        "Do NOT repeat yourself. Be factual. Max 120 words."
    )})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_BATCH,
            repetition_penalty=1.2,
            no_repeat_ngram_size=6,
        )
    response = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    del inputs, output
    torch.cuda.empty_cache()
    return response


def analyze_video(video_path, model_id, whisper_model="turbo", fps=4,
                  batch_size=64, hint="", no_audio=False, verbose=False):
    """Full video analysis pipeline. Returns structured result dict."""
    import torch

    timings = {}
    total_start = time.time()

    # Stage 1: Extract
    if verbose:
        print("[1/4] Extracting frames and audio...")
    t0 = time.time()
    frames, duration = extract_frames(video_path, fps=fps)
    wav_path = None if no_audio else extract_audio(video_path)
    timings['extract'] = time.time() - t0
    if verbose:
        print(f"  {len(frames)} frames, {fmt_time(duration)}")

    # Stage 2: Transcribe
    transcript = None
    language = None
    if wav_path:
        if verbose:
            print(f"[2/4] Transcribing with Whisper {whisper_model}...")
        t0 = time.time()
        transcript, language = transcribe(wav_path, model_size=whisper_model, verbose=verbose)
        timings['whisper'] = time.time() - t0
        os.unlink(wav_path)
        if verbose:
            print(f"  {len(transcript)} segments, lang={language}, {timings['whisper']:.1f}s")
    else:
        timings['whisper'] = 0

    # Stage 3: Visual analysis
    if verbose:
        print(f"[3/4] Analyzing with {model_id}...")
    t0 = time.time()
    model, processor = load_vlm(model_id, verbose=verbose)
    hint_text = f"This is a demo/video for '{hint}'. " if hint else ""

    descriptions = []
    total_batches = (len(frames) + batch_size - 1) // batch_size
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_num = i // batch_size + 1
        batch_start = batch[0][0]
        batch_end = batch[-1][0]
        batch_pil = [f[1] for f in batch]

        if verbose:
            print(f"  Batch {batch_num}/{total_batches} [{fmt_time(batch_start)}-{fmt_time(batch_end)}]")

        try:
            desc = analyze_batch(model, processor, batch_pil, batch_start, batch_end, hint_text)
        except torch.cuda.OutOfMemoryError:
            half = len(batch_pil) // 2
            if verbose:
                print(f"  [!] OOM, retrying with {half} frames")
            torch.cuda.empty_cache()
            desc = analyze_batch(model, processor, batch_pil[:half], batch_start,
                                 batch[half-1][0], hint_text)

        descriptions.append({"start": batch_start, "end": batch_end, "description": desc})

    timings['vlm'] = time.time() - t0

    # Stage 4: Synthesis
    if verbose:
        print("[4/4] Synthesizing...")
    t0 = time.time()
    synthesis = _synthesize(model, processor, descriptions, transcript)
    timings['synthesis'] = time.time() - t0

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    timings['total'] = time.time() - total_start

    return {
        "video_path": video_path,
        "duration": duration,
        "frames_analyzed": len(frames),
        "language": language,
        "timings": timings,
        "visual_timeline": descriptions,
        "transcript": transcript,
        "synthesis": synthesis,
    }


def _synthesize(model, processor, visual_descs, transcript):
    """Generate final synthesis from visual descriptions + transcript."""
    import torch

    visual_text = "\n".join(
        f"[{fmt_time(d['start'])}-{fmt_time(d['end'])}] {d['description']}"
        for d in visual_descs
    )
    transcript_text = "\n".join(
        f"[{fmt_time(s['start'])}] \"{s['text']}\""
        for s in (transcript or [])
    )

    prompt = f"""Analyze this video. Be concise, no filler.

1. SUMMARY (2-3 sentences): What is shown? What's the main message?
2. KEY MOMENTS (2-3 with timestamps): Most impactful visual moments. Give exact timestamps and describe the visual hook.
3. STRENGTHS: What works well visually/narratively?
4. WEAKNESSES: What could be improved?

VISUAL TIMELINE:
{visual_text}

TRANSCRIPT:
{transcript_text if transcript_text else "(no audio)"}
"""
    if len(prompt) > 30000:
        prompt = prompt[:30000] + "\n...(truncated)"

    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_SYNTHESIS,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )
    response = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    del inputs, output
    torch.cuda.empty_cache()
    return response

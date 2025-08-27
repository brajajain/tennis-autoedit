import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

import numpy as np
from pydub import AudioSegment, effects
from pydub.effects import low_pass_filter, high_pass_filter
from rapidfuzz import fuzz, process
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ---------- utils ----------
def srt_ts(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - math.floor(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def clean_audio(in_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load with pydub/ffmpeg, mono, 16 kHz, speech-band filters, normalize."""
    seg = AudioSegment.from_file(in_path)
    seg = seg.set_channels(1).set_frame_rate(target_sr)
    seg = effects.strip_silence(seg, silence_len=200, silence_thresh=seg.dBFS - 16, padding=100)
    seg = high_pass_filter(seg, cutoff=100)
    seg = low_pass_filter(seg, cutoff=4000)
    seg = effects.normalize(seg)
    # to float32 -1..1
    arr = np.array(seg.get_array_of_samples()).astype(np.float32)
    arr /= np.iinfo(seg.array_type).max
    return arr, target_sr

def transcribe_whisper(samples: np.ndarray, sr: int, model_size: str = "tiny.en",
                       device: str = "cpu", compute_type: str = "int8") -> list[dict]:
    """
    Returns [{'word': str, 'start': float, 'end': float, 'conf': float}, ...]
    using faster-whisper with per-word timestamps.
    """
    # Write a temporary WAV in 16k mono for simplicity (pydub already normalized/filtered)
    import tempfile, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, samples, sr)
        tmp_path = tmp.name

    # On Apple Silicon you can also try device="metal" (hardware accel) if you want
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        tmp_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
    )
    words: list[dict] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": float(w.start or seg.start),
                    "end": float(w.end or seg.end),
                    "conf": float(getattr(w, "probability", 0.0)),
                })
        else:
            # fallback if the model didn't emit word pieces for a segment
            words.append({
                "word": seg.text.strip(),
                "start": float(seg.start),
                "end": float(seg.end),
                "conf": 0.0,
            })
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return words


def write_srt(words: List[Dict], path: str) -> None:
    if not words:
        Path(path).write_text("")
        return
    entries = []
    chunk = []
    start_t = words[0]["start"]
    last_t = words[0]["end"]
    MAX_CHUNK = 5.0
    for w in words:
        if (w["end"] - start_t) > MAX_CHUNK:
            text = " ".join(x["word"] for x in chunk)
            entries.append((start_t, last_t, text))
            chunk = [w]; start_t = w["start"]; last_t = w["end"]
        else:
            chunk.append(w); last_t = w["end"]
    if chunk:
        text = " ".join(x["word"] for x in chunk)
        entries.append((start_t, last_t, text))
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(entries, 1):
            f.write(f"{i}\n{srt_ts(a)} --> {srt_ts(b)}\n{text.strip()}\n\n")

# ---------- multi-segment logic ----------
KEY_CANON = {
    "start": {
        "start", "let's start", "begin", "go", "mark start", "start now",
        "okay start", "ok start", "and start", "start it", "we start"
    },
    "stop": {
        "stop", "end", "that's it", "mark stop", "stop now", "finish",
        "we're done", "all done", "and stop"
    },
}

def _window_phrases(tokens: List[str], i: int, n: int) -> str:
    return " ".join(tokens[i:i+n])

def find_all_keyword_times(words: List[Dict], target: str, thresh: int) -> List[float]:
    canon = list(KEY_CANON.get(target, {target}))
    tokens = [w["word"].lower() for w in words]
    times = [w["start"] for w in words]
    hits: List[float] = []
    for i, tok in enumerate(tokens):
        if fuzz.ratio(tok, target) >= thresh:
            hits.append(times[i])
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            phrase = _window_phrases(tokens, i, n)
            match, score, _ = process.extractOne(phrase, canon, scorer=fuzz.ratio)
            if score >= thresh:
                hits.append(times[i])
    return sorted(set(hits))

def pair_segments(starts: List[float], stops: List[float], default_span: float) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    starts = sorted(starts)
    stops = sorted(stops)
    si, ei = 0, 0
    open_start: Optional[float] = None
    while si < len(starts) or ei < len(stops):
        next_start = starts[si] if si < len(starts) else float("inf")
        next_stop  = stops[ei] if ei < len(stops) else float("inf")
        if open_start is None:
            if next_start < next_stop:
                open_start = next_start
                si += 1
            else:
                ei += 1
        else:
            if next_stop < next_start:
                if next_stop > open_start:
                    segments.append((open_start, next_stop))
                open_start = None
                ei += 1
            else:
                segments.append((open_start, open_start + default_span))
                open_start = next_start
                si += 1
    if open_start is not None:
        segments.append((open_start, open_start + default_span))
    segments = sorted(set((round(a, 3), round(b, 3)) for a, b in segments if b > a))
    return segments

def expand_and_merge(segments: List[Tuple[float, float]],
                     pad_pre: float, pad_post: float, merge_gap: float,
                     absolute_end: Optional[float] = None) -> List[Tuple[float, float]]:
    if not segments:
        return []
    padded = []
    for s, e in segments:
        a = s - pad_pre
        b = e + pad_post
        if absolute_end is not None:
            a = max(0.0, a)
            b = min(absolute_end, b)
        padded.append((a, b))
    padded.sort()
    merged = [padded[0]]
    for s, e in padded[1:]:
        ps, pe = merged[-1]
        if s <= pe + merge_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def stitch_audio_segments(orig_path: str, segments: List[Tuple[float, float]],
                         out_path: str, crossfade_ms: int = 0,
                         chunks_dir: Optional[str] = None) -> None:
    src = AudioSegment.from_file(orig_path).set_channels(1)
    out = AudioSegment.silent(duration=0, frame_rate=src.frame_rate)
    if chunks_dir:
        Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    for i, (a, b) in enumerate(segments):
        seg = src[int(a * 1000):int(b * 1000)]
        if chunks_dir:
            seg.export(Path(chunks_dir) / f"chunk_{i:03d}.wav", format="wav")
        if len(out) == 0 or crossfade_ms <= 0:
            out += seg
        else:
            out = out.append(seg, crossfade=crossfade_ms)
    ext = Path(out_path).suffix.replace(".", "") or "wav"
    out.export(out_path, format=ext)

def write_segments_csv(segments: List[Tuple[float, float]], csv_path: str) -> None:
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "start_sec", "end_sec", "duration_sec"])
        for i, (a, b) in enumerate(segments):
            w.writerow([i, f"{a:.3f}", f"{b:.3f}", f"{b - a:.3f}"])

# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Trim multiple segments between spoken start/stop markers; emit stitched audio + SRT + CSV.")
    ap.add_argument("--in_audio", required=True, help="Path to memo (m4a/mp3/wav)")
    ap.add_argument("--out_audio", default="cuts.wav", help="Stitched output audio")
    ap.add_argument("--out_srt", default="memo.srt", help="Full-memo transcript SRT (for debugging)")
    ap.add_argument("--segments_csv", default="segments.csv", help="CSV with segment timings")
    ap.add_argument("--chunks_dir", default=None, help="Optional: export each segment as its own WAV in this directory")
    ap.add_argument("--start_kw", default="start")
    ap.add_argument("--stop_kw", default="stop")
    ap.add_argument("--fuzzy", type=int, default=85, help="Fuzzy match threshold (0-100)")
    ap.add_argument("--default_span", type=float, default=8.0, help="If a start lacks a stop, duration to use")
    ap.add_argument("--pad_pre", type=float, default=0.5, help="Seconds to pad before each segment")
    ap.add_argument("--pad_post", type=float, default=0.5, help="Seconds to pad after each segment")
    ap.add_argument("--merge_gap", type=float, default=0.3, help="Merge segments with gaps <= this (sec)")
    ap.add_argument("--crossfade_ms", type=int, default=0, help="Crossfade when stitching (ms)")
    ap.add_argument("--model", default="tiny.en", help="faster-whisper model size (e.g., tiny.en, base.en)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "metal"], help="Inference device")
    ap.add_argument("--compute_type", default="int8", help="CTranslate2 compute type (e.g., int8, int8_float16, float16)")
    args = ap.parse_args()

    samples, sr = clean_audio(args.in_audio, target_sr=16000)
    words = transcribe_whisper(samples, sr, args.model, args.device, args.compute_type)
    write_srt(words, args.out_srt)

    start_times = find_all_keyword_times(words, args.start_kw.lower(), args.fuzzy)
    stop_times  = find_all_keyword_times(words, args.stop_kw.lower(),  args.fuzzy)

    if not start_times and not stop_times:
        print("No start/stop keywords detected. Nothing to cut.")
        return

    raw_segments = pair_segments(start_times, stop_times, args.default_span)
    absolute_end = float(len(AudioSegment.from_file(args.in_audio))) / 1000.0
    segments = expand_and_merge(raw_segments, args.pad_pre, args.pad_post, args.merge_gap, absolute_end=absolute_end)

    if not segments:
        print("No valid segments after pairing/merging.")
        return

    stitch_audio_segments(
        orig_path=args.in_audio,
        segments=segments,
        out_path=args.out_audio,
        crossfade_ms=args.crossfade_ms,
        chunks_dir=args.chunks_dir,
    )

    write_segments_csv(segments, args.segments_csv)

    total_dur = sum(b - a for a, b in segments)
    print(f"Segments: {len(segments)} | Total kept: {total_dur:.2f}s → {args.out_audio}")
    print(f"SRT: {args.out_srt}")
    print(f"CSV: {args.segments_csv}")
    if args.chunks_dir:
        print(f"Chunks: {args.chunks_dir}")

if __name__ == "__main__":
    main()

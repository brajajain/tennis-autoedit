"""Audio processing utilities for tennis-autoedit."""
from pathlib import Path
from typing import Tuple

import numpy as np
from pydub import AudioSegment, effects
from pydub.effects import low_pass_filter, high_pass_filter

def clean_audio(in_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
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

def stitch_audio_segments(
    orig_path: str, 
    segments: list[tuple[float, float]],
    out_path: str, 
    crossfade_ms: int = 0,
    chunks_dir: str | None = None
) -> None:
    """Stitch audio segments together with optional crossfading."""
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

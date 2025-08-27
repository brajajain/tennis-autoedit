"""Transcription functionality using Whisper for speech-to-text."""
import os
import tempfile
from typing import List, Dict

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

def transcribe_whisper(
    samples: np.ndarray, 
    sr: int, 
    model_size: str = "tiny.en",
    device: str = "cpu", 
    compute_type: str = "int8"
) -> List[Dict]:
    """
    Transcribe audio using Whisper with word-level timestamps.
    
    Returns:
        List of dicts with keys: word, start, end, conf
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, samples, sr)
        tmp_path = tmp.name

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(
        tmp_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 250},
    )
    
    words: List[Dict] = []
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

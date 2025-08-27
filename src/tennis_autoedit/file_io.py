"""File I/O operations for tennis-autoedit."""
import csv
import math
from pathlib import Path
from typing import List, Dict, Tuple

def srt_ts(sec: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - math.floor(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(words: List[Dict], path: str) -> None:
    """Write words with timestamps to SRT file."""
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
            chunk = [w]
            start_t = w["start"]
            last_t = w["end"]
        else:
            chunk.append(w)
            last_t = w["end"]
    
    if chunk:
        text = " ".join(x["word"] for x in chunk)
        entries.append((start_t, last_t, text))
    
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(entries, 1):
            f.write(f"{i}\n{srt_ts(a)} --> {srt_ts(b)}\n{text.strip()}\n\n")

def write_segments_csv(segments: List[Tuple[float, float]], csv_path: str) -> None:
    """Write segment information to a CSV file."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "start_sec", "end_sec", "duration_sec"])
        for i, (a, b) in enumerate(segments):
            w.writerow([i, f"{a:.3f}", f"{b:.3f}", f"{b - a:.3f}"])

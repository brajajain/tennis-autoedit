"""Segment detection and processing logic."""
from typing import List, Tuple, Optional, Dict, Any
from rapidfuzz import fuzz, process

# Default keyword sets for segment detection
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
    """Helper to extract a phrase from tokens."""
    return " ".join(tokens[i:i+n])

def find_all_keyword_times(
    words: List[Dict[str, Any]], 
    target: str, 
    threshold: int
) -> List[float]:
    """Find all occurrences of keyword in transcription with fuzzy matching."""
    canon = list(KEY_CANON.get(target, {target}))
    tokens = [w["word"].lower() for w in words]
    times = [w["start"] for w in words]
    hits: List[float] = []
    
    # Single word matches
    for i, tok in enumerate(tokens):
        if fuzz.ratio(tok, target) >= threshold:
            hits.append(times[i])
    
    # Multi-word phrase matches
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            phrase = _window_phrases(tokens, i, n)
            match, score, _ = process.extractOne(phrase, canon, scorer=fuzz.ratio)
            if score >= threshold:
                hits.append(times[i])
    
    return sorted(set(hits))

def pair_segments(
    starts: List[float], 
    stops: List[float], 
    default_span: float
) -> List[Tuple[float, float]]:
    """Pair start and stop times into segments."""
    segments: List[Tuple[float, float]] = []
    starts = sorted(starts)
    stops = sorted(stops)
    si, ei = 0, 0
    open_start: Optional[float] = None
    
    while si < len(starts) or ei < len(stops):
        next_start = starts[si] if si < len(starts) else float("inf")
        next_stop = stops[ei] if ei < len(stops) else float("inf")
        
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
    
    return sorted(set((round(a, 3), round(b, 3)) for a, b in segments if b > a))

def expand_and_merge(
    segments: List[Tuple[float, float]],
    pad_pre: float, 
    pad_post: float, 
    merge_gap: float,
    absolute_end: Optional[float] = None
) -> List[Tuple[float, float]]:
    """Apply padding and merge overlapping/close segments."""
    if not segments:
        return []
    
    # Apply padding
    padded = []
    for s, e in segments:
        a = s - pad_pre
        b = e + pad_post
        if absolute_end is not None:
            a = max(0.0, a)
            b = min(absolute_end, b)
        padded.append((a, b))
    
    # Sort and merge
    padded.sort()
    merged = [padded[0]]
    
    for s, e in padded[1:]:
        ps, pe = merged[-1]
        if s <= pe + merge_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    
    return merged

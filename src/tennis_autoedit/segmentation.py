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

def smooth_vlm_predictions(predictions: List[bool]) -> List[bool]:
    """
    Bridge [True, False, True] sequences to maintain continuous rally clips.
    """
    if len(predictions) < 3:
        return predictions

    smoothed = list(predictions)
    for i in range(1, len(predictions) - 1):
        if predictions[i-1] is True and predictions[i+1] is True:
            smoothed[i] = True

    return smoothed

def get_video_segments(
    predictions: List[bool],
    chunk_duration: float,
    pre_buffer: float = 5.0,
    post_buffer: float = 3.0,
    min_duration: float = 2.0,
    absolute_end: Optional[float] = None
) -> List[Tuple[float, float]]:
    """
    Convert chunk-based predictions into time segments with buffers.
    """
    raw_segments: List[Tuple[float, float]] = []
    in_segment = False
    start_time = 0.0

    for i, active in enumerate(predictions):
        current_time = i * chunk_duration
        if active and not in_segment:
            start_time = current_time
            in_segment = True
        elif not active and in_segment:
            end_time = current_time # end of previous chunk
            raw_segments.append((start_time, end_time))
            in_segment = False

    if in_segment:
        raw_segments.append((start_time, len(predictions) * chunk_duration))

    # Apply buffers and merge
    return expand_and_merge(
        raw_segments,
        pad_pre=pre_buffer,
        pad_post=post_buffer,
        merge_gap=0.1,
        absolute_end=absolute_end
    )

def filter_short_segments(
    segments: List[Tuple[float, float]],
    min_duration: float
) -> List[Tuple[float, float]]:
    """
    Remove segments that are shorter than min_duration.
    """
    return [s for s in segments if (s[1] - s[0]) >= min_duration]

"""Command-line interface for tennis-autoedit."""
import argparse
from pathlib import Path
from typing import Optional

from pydub import AudioSegment

from . import audio_utils, file_io, segmentation, transcription

def main() -> None:
    """Main entry point for the CLI."""
    ap = argparse.ArgumentParser(
        description="Trim multiple segments between spoken start/stop markers; "
                    "emit stitched audio + SRT + CSV."
    )
    # Input/output files
    ap.add_argument("--in_audio", required=True, 
                   help="Path to input audio (m4a/mp3/wav)")
    ap.add_argument("--out_audio", default="cuts.wav", 
                   help="Stitched output audio")
    ap.add_argument("--out_srt", default="memo.srt", 
                   help="Full-memo transcript SRT (for debugging)")
    ap.add_argument("--segments_csv", default="segments.csv", 
                   help="CSV with segment timings")
    ap.add_argument("--chunks_dir", default=None, 
                   help="Optional: export each segment as its own WAV in this directory")
    
    # Segmentation parameters
    ap.add_argument("--start_kw", default="start", 
                   help="Keyword to detect segment starts")
    ap.add_argument("--stop_kw", default="stop", 
                   help="Keyword to detect segment ends")
    ap.add_argument("--fuzzy", type=int, default=85, 
                   help="Fuzzy match threshold (0-100)")
    ap.add_argument("--default_span", type=float, default=8.0, 
                   help="If a start lacks a stop, duration to use (seconds)")
    ap.add_argument("--pad_pre", type=float, default=0.5, 
                   help="Seconds to pad before each segment")
    ap.add_argument("--pad_post", type=float, default=0.5, 
                   help="Seconds to pad after each segment")
    ap.add_argument("--merge_gap", type=float, default=0.3, 
                   help="Merge segments with gaps <= this (seconds)")
    ap.add_argument("--crossfade_ms", type=int, default=0, 
                   help="Crossfade when stitching (milliseconds)")
    
    # Whisper model parameters
    ap.add_argument("--model", default="tiny.en", 
                   help="faster-whisper model size (e.g., tiny.en, base.en)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "metal"], 
                   help="Inference device")
    ap.add_argument("--compute_type", default="int8", 
                   help="CTranslate2 compute type (e.g., int8, int8_float16, float16)")
    
    args = ap.parse_args()

    # 1. Process audio and transcribe
    samples, sr = audio_utils.clean_audio(args.in_audio, target_sr=16000)
    words = transcription.transcribe_whisper(
        samples, sr, 
        model_size=args.model, 
        device=args.device, 
        compute_type=args.compute_type
    )
    
    # 2. Generate SRT file
    file_io.write_srt(words, args.out_srt)

    # 3. Find segment boundaries
    start_times = segmentation.find_all_keyword_times(
        words, args.start_kw.lower(), args.fuzzy
    )
    stop_times = segmentation.find_all_keyword_times(
        words, args.stop_kw.lower(), args.fuzzy
    )

    if not start_times and not stop_times:
        print("No start/stop keywords detected. Nothing to cut.")
        return

    # 4. Process segments
    raw_segments = segmentation.pair_segments(
        start_times, stop_times, args.default_span
    )
    
    # Get audio duration in seconds
    audio = AudioSegment.from_file(args.in_audio)
    absolute_end = len(audio) / 1000.0
    
    segments = segmentation.expand_and_merge(
        raw_segments, 
        args.pad_pre, 
        args.pad_post, 
        args.merge_gap, 
        absolute_end=absolute_end
    )

    if not segments:
        print("No valid segments after pairing/merging.")
        return

    # 5. Process and output audio
    audio_utils.stitch_audio_segments(
        orig_path=args.in_audio,
        segments=segments,
        out_path=args.out_audio,
        crossfade_ms=args.crossfade_ms,
        chunks_dir=args.chunks_dir,
    )

    # 6. Write segments CSV
    file_io.write_segments_csv(segments, args.segments_csv)

    # 7. Print summary
    total_dur = sum(b - a for a, b in segments)
    print(f"Segments: {len(segments)} | Total kept: {total_dur:.2f}s → {args.out_audio}")
    print(f"SRT: {args.out_srt}")
    print(f"CSV: {args.segments_csv}")
    if args.chunks_dir:
        print(f"Chunks: {args.chunks_dir}")


if __name__ == "__main__":
    main()

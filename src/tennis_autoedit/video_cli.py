"""CLI for VLM-based tennis video editing."""
import argparse
from typing import List

from . import vlm_logic, video_utils, segmentation

def main() -> None:
    """Main entry point for the video CLI."""
    ap = argparse.ArgumentParser(
        description="Extract 'Active Play' segments from a tennis video using VLM."
    )
    ap.add_argument("--video", required=True, help="Path to input video file")
    ap.add_argument("--output", default="active_play.mp4", help="Path to output video file")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
                   help="MLX VLM model path")
    ap.add_argument("--chunk_duration", type=float, default=5.0,
                   help="Duration of each video chunk to analyze (seconds)")
    ap.add_argument("--fps", type=float, default=1.0,
                   help="Frames per second to extract from each chunk for the VLM")
    ap.add_argument("--pre_buffer", type=float, default=5.0,
                   help="Seconds to pad before a detected segment")
    ap.add_argument("--post_buffer", type=float, default=3.0,
                   help="Seconds to pad after a detected segment")
    ap.add_argument("--min_duration", type=float, default=2.0,
                   help="Minimum duration for a segment to be kept")

    args = ap.parse_args()

    # 1. Load VLM
    print(f"Loading VLM model: {args.model}...")
    model, processor, config = vlm_logic.load_vlm(args.model)

    # 2. Process video in chunks
    video_duration = video_utils.get_video_duration(args.video)
    print(f"Processing video: {args.video} ({video_duration:.2f}s)")

    predictions: List[bool] = []
    num_chunks = int(video_duration / args.chunk_duration)

    for i in range(num_chunks):
        start_time = i * args.chunk_duration
        print(f"Analyzing chunk {i+1}/{num_chunks} (at {start_time:.2f}s)...", end="\r")

        # Extract frames for the chunk
        images = video_utils.extract_frames_from_chunk(
            args.video, start_time, args.chunk_duration, fps=args.fps
        )

        if not images:
            predictions.append(False)
            continue

        # VLM Inference
        is_active = vlm_logic.check_point_in_progress(
            model, processor, config, images
        )
        predictions.append(is_active)

    print("\nAnalysis complete.")

    # 3. Temporal Smoothing
    smoothed_predictions = segmentation.smooth_vlm_predictions(predictions)

    # 4. Generate Segments
    segments = segmentation.get_video_segments(
        smoothed_predictions,
        args.chunk_duration,
        pre_buffer=args.pre_buffer,
        post_buffer=args.post_buffer,
        absolute_end=video_duration
    )

    # 5. Filter short segments (False Positive strategy)
    filtered_segments = segmentation.filter_short_segments(segments, args.min_duration)

    print(f"Found {len(filtered_segments)} active play segments.")
    for i, (s, e) in enumerate(filtered_segments):
        print(f"  Segment {i+1}: {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")

    # 6. Assemble Video
    if filtered_segments:
        print(f"Assembling final video to {args.output}...")
        video_utils.assemble_video_segments(args.video, filtered_segments, args.output)
        print("Done.")
    else:
        print("No active play segments found. Output video not created.")

if __name__ == "__main__":
    main()

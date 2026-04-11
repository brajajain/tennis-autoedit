import cv2
import subprocess
from PIL import Image
from typing import List, Tuple
import os

def extract_frames_from_chunk(
    video_path: str,
    start_time: float,
    duration: float,
    fps: float = 1.0
) -> List[Image.Image]:
    """
    Extract frames from a specific chunk of the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * video_fps)
    num_frames_to_extract = int(duration * fps)

    # We want to space out the frames evenly within the chunk
    frame_indices = [
        start_frame + int(i * (video_fps / fps))
        for i in range(num_frames_to_extract)
    ]

    images = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame_rgb))
        else:
            break

    cap.release()
    return images

def assemble_video_segments(
    video_path: str,
    segments: List[Tuple[float, float]],
    output_path: str
) -> None:
    """
    Assemble the final video from segments using FFmpeg with h264_videotoolbox.
    Uses a concat demuxer for efficiency with many segments.
    """
    if not segments:
        print("No segments to assemble.")
        return

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        list_file = os.path.join(tmpdir, "concat_list.txt")
        segment_files = []

        for i, (start, end) in enumerate(segments):
            seg_file = os.path.join(tmpdir, f"seg_{i}.mp4")
            duration = end - start

            # Extract segment with fast seeking
            cmd_extract = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(duration),
                "-c:v", "h264_videotoolbox",
                "-b:v", "5000k",
                "-c:a", "aac",
                seg_file
            ]
            subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            segment_files.append(seg_file)

        with open(list_file, "w") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")

        # Concat segments
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]

        print(f"Assembling {len(segments)} segments...")
        subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_video_duration(video_path: str) -> float:
    """
    Get the duration of the video in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration

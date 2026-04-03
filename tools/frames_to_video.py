#!/usr/bin/env python3
"""Create a video file from an image frame directory.

Examples:
    python tools/frames_to_video.py --frames-dir data/realsense/demo
    python tools/frames_to_video.py --frames-dir data/realsense/demo --output data/realsense/demo.mp4
    python tools/frames_to_video.py --frames-dir data/realsense/demo --fps 15 --codec XVID --output data/realsense/demo.avi
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an mp4 or avi video from a directory of image frames."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing RGB frames such as 000000.png or YYYYMMDD_HHMMSS_mmm.png.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to <frames-dir>.mp4 next to the frame directory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video FPS. If omitted, the tool tries frames-dir/recording_metadata.json and falls back to 30.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default=None,
        help="Optional FOURCC codec override. Defaults to mp4v for .mp4 and XVID for .avi.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern used to select frame files inside --frames-dir. Default: %(default)s",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def choose_codec(output_path: Path, codec_override: str | None) -> str:
    if codec_override:
        if len(codec_override) != 4:
            raise SystemExit("--codec must be exactly four characters, e.g. mp4v or XVID.")
        return codec_override

    if output_path.suffix.lower() == ".avi":
        return "XVID"
    return "mp4v"


def resolve_output_path(frames_dir: Path, output: Path | None) -> Path:
    if output is None:
        return frames_dir.with_suffix(".mp4").resolve()

    output_path = output.expanduser()
    if output_path.suffix:
        return output_path.resolve()
    return output_path.with_suffix(".mp4").resolve()


def resolve_fps(frames_dir: Path, fps_override: float | None) -> tuple[float, Path | None]:
    if fps_override is not None:
        if fps_override <= 0:
            raise SystemExit("--fps must be greater than 0.")
        return fps_override, None

    metadata_path = frames_dir / "recording_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse metadata file {metadata_path}: {exc}") from exc

        for key in ("requested_fps", "fps"):
            value = metadata.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value), metadata_path

    return DEFAULT_FPS, metadata_path if metadata_path.exists() else None


def collect_frame_paths(frames_dir: Path, pattern: str) -> list[Path]:
    frame_paths = [
        path
        for path in frames_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    frame_paths.sort(key=natural_sort_key)
    return frame_paths


def load_frame(frame_path: Path) -> "cv2.typing.MatLike":
    image = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Failed to read frame: {frame_path}")

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image
    raise SystemExit(
        f"Unsupported frame shape {image.shape} for {frame_path}. Expected grayscale, BGR, or BGRA."
    )


def main() -> int:
    args = parse_args()
    frames_dir = args.frames_dir.expanduser().resolve()
    if not frames_dir.is_dir():
        raise SystemExit(f"Frame directory does not exist: {frames_dir}")

    output_path = resolve_output_path(frames_dir, args.output)
    fps, fps_source = resolve_fps(frames_dir, args.fps)
    frame_paths = collect_frame_paths(frames_dir, args.pattern)
    if not frame_paths:
        raise SystemExit(
            f"No image frames found in {frames_dir} matching pattern {args.pattern!r} and supported suffixes "
            f"{sorted(SUPPORTED_IMAGE_SUFFIXES)}."
        )

    first_frame = load_frame(frame_paths[0])
    frame_size = (first_frame.shape[1], first_frame.shape[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = choose_codec(output_path, args.codec)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise SystemExit(
            f"Failed to open video writer for {output_path}. Try --codec XVID and an .avi output."
        )

    frames_written = 0
    try:
        for frame_path in frame_paths:
            frame = load_frame(frame_path)
            current_size = (frame.shape[1], frame.shape[0])
            if current_size != frame_size:
                raise SystemExit(
                    f"Frame size mismatch for {frame_path}: expected {frame_size[0]}x{frame_size[1]}, "
                    f"got {current_size[0]}x{current_size[1]}."
                )
            writer.write(frame)
            frames_written += 1
    finally:
        writer.release()

    print(f"Saved video: {output_path}")
    print(f"Frames written: {frames_written}")
    print(f"FPS: {fps:g}")
    print(f"Codec: {codec}")
    if fps_source is not None and args.fps is None:
        print(f"FPS source: {fps_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

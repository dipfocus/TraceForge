#!/usr/bin/env python3
"""Record RGB video or frame folders from an Intel RealSense D435i camera.

Examples:
    python tools/record_realsense.py
    python tools/record_realsense.py --mode video --output data/realsense/demo.mp4 --preview --show-depth
    python tools/record_realsense.py --frames-dir data/realsense/demo --save-depth
    python tools/record_realsense.py --list-devices
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

rs = None
DEPTH_PNG_SCALE = 10000.0
FRAME_IMAGE_FORMAT = "png"


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = Path("data") / timestamp

    parser = argparse.ArgumentParser(
        description="Record color video or frame folders from an Intel RealSense D435i camera."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Base output path. Defaults to a timestamp-based path under data/ and is used to derive the video path and/or frame directories.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="frames",
        choices=("video", "frames", "both"),
        help="Save video, frame directory, or both. Default: %(default)s",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory to save infer.py-compatible image frames such as YYYYMMDD_HHMMSS_mmm.png. If omitted, a timestamp-based directory is derived from --output. Passing data/ creates data/<timestamp>/ automatically.",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="Save depth frames aligned to the RGB stream for infer.py --depth_path. Depth PNGs are written to a depth/ subdirectory under the resolved RGB frame directory, for example data/<timestamp>/depth.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds. Use 0 to record until Ctrl-C (or q in preview).",
    )
    parser.add_argument("--width", type=int, default=1280, help="Color stream width.")
    parser.add_argument("--height", type=int, default=720, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS.")
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Optional device serial number when multiple RealSense cameras are connected.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default=None,
        help="Optional FOURCC codec override. Defaults to one based on the output suffix.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to discard before recording so auto exposure can settle.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live OpenCV preview while recording.",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show aligned depth next to RGB in the preview window. Requires --preview.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List connected RealSense devices and exit.",
    )
    args = parser.parse_args()
    args.recording_timestamp = timestamp
    return args


def require_realsense():
    import pyrealsense2 as realsense
    return realsense


def get_camera_info(device: "rs.device", info_key: "rs.camera_info") -> str | None:
    if device.supports(info_key):
        return device.get_info(info_key)
    return None


def list_devices() -> list[dict[str, str | None]]:
    ctx = rs.context()
    devices = []
    for device in ctx.devices:
        devices.append(
            {
                "name": get_camera_info(device, rs.camera_info.name),
                "serial": get_camera_info(device, rs.camera_info.serial_number),
                "firmware": get_camera_info(device, rs.camera_info.firmware_version),
            }
        )
    return devices


def print_devices(devices: list[dict[str, str | None]]) -> None:
    if not devices:
        print("No RealSense device detected.")
        return

    for idx, device in enumerate(devices, start=1):
        print(
            f"[{idx}] name={device['name']} serial={device['serial']} firmware={device['firmware']}"
        )


def ensure_requested_device_exists(serial: str | None) -> None:
    devices = list_devices()
    if not devices:
        raise SystemExit("No RealSense device detected.")

    if serial is None:
        return

    available_serials = {device["serial"] for device in devices}
    if serial not in available_serials:
        print_devices(devices)
        raise SystemExit(f"Requested serial '{serial}' was not found among connected devices.")


def choose_codec(output_path: Path, codec_override: str | None) -> str:
    if codec_override:
        if len(codec_override) != 4:
            raise SystemExit("--codec must be exactly four characters, e.g. mp4v or XVID.")
        return codec_override

    suffix = output_path.suffix.lower()
    if suffix == ".avi":
        return "XVID"
    return "mp4v"


def resolve_depth_dir(
    args: argparse.Namespace,
    frames_dir: Path | None,
) -> Path | None:
    if not args.save_depth:
        return None

    if frames_dir is not None:
        return frames_dir / "depth"

    output_path = args.output.expanduser()
    if output_path.suffix:
        return output_path.with_suffix("").resolve() / "depth"
    return output_path.resolve() / "depth"


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path]:
    save_video = args.mode in {"video", "both"}
    save_frames = args.mode in {"frames", "both"}
    recording_timestamp = getattr(
        args,
        "recording_timestamp",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    output_path = args.output.expanduser()
    if save_video:
        video_output = output_path if output_path.suffix else output_path.with_suffix(".mp4")
        video_path = video_output.resolve()
    else:
        video_path = None

    if save_frames:
        if args.frames_dir is not None:
            requested_frames_dir = args.frames_dir.expanduser().resolve()
            project_data_dir = Path("data").resolve()
            if requested_frames_dir == project_data_dir:
                frames_dir = requested_frames_dir / recording_timestamp
            else:
                frames_dir = requested_frames_dir
        elif output_path.suffix:
            frames_dir = output_path.expanduser().with_suffix("").resolve()
        else:
            frames_dir = output_path.expanduser().resolve()
    else:
        frames_dir = None

    if video_path is not None:
        metadata_path = video_path.with_suffix(".json")
    else:
        metadata_path = frames_dir / "recording_metadata.json"

    return video_path, frames_dir, metadata_path


def format_frame_stem(timestamp_ns: int) -> str:
    timestamp_dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000)
    return timestamp_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def next_frame_timestamp_ns(last_timestamp_ns: int | None) -> int:
    timestamp_ns = time.time_ns()
    if last_timestamp_ns is None:
        return timestamp_ns
    return max(timestamp_ns, last_timestamp_ns + 1_000_000)


def save_frame(
    image: np.ndarray,
    frame_stem: str,
    frames_dir: Path,
) -> Path:
    frame_path = frames_dir / f"{frame_stem}.{FRAME_IMAGE_FORMAT}"
    ok = cv2.imwrite(str(frame_path), image)
    if not ok:
        raise SystemExit(f"Failed to write frame to {frame_path}")
    return frame_path


def save_depth_frame(
    depth_meters: np.ndarray,
    frame_stem: str,
    depth_dir: Path,
) -> bool:
    encoded_depth = np.clip(
        np.rint(depth_meters * DEPTH_PNG_SCALE),
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    depth_path = depth_dir / f"{frame_stem}.png"
    ok = cv2.imwrite(str(depth_path), encoded_depth)
    if not ok:
        raise SystemExit(f"Failed to write depth frame to {depth_path}")
    return bool(np.any(depth_meters * DEPTH_PNG_SCALE > np.iinfo(np.uint16).max))


def intrinsics_to_dict(intrinsics) -> dict[str, object]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def build_metadata(
    profile: "rs.pipeline_profile",
    include_depth: bool = False,
    depth_scale: float | None = None,
) -> dict[str, object]:
    device = profile.get_device()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = color_stream.get_intrinsics()

    metadata = {
        "device_name": get_camera_info(device, rs.camera_info.name),
        "device_serial": get_camera_info(device, rs.camera_info.serial_number),
        "firmware_version": get_camera_info(device, rs.camera_info.firmware_version),
        "color_intrinsics": intrinsics_to_dict(color_intrinsics),
    }

    if include_depth:
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrinsics = depth_stream.get_intrinsics()
        metadata["depth_intrinsics"] = intrinsics_to_dict(depth_intrinsics)
        metadata["aligned_depth_intrinsics"] = intrinsics_to_dict(color_intrinsics)
        metadata["depth_scale_meters_per_unit"] = depth_scale

    return metadata


def main() -> int:
    global rs
    args = parse_args()
    rs = require_realsense()

    if args.list_devices:
        print_devices(list_devices())
        return 0

    ensure_requested_device_exists(args.serial)

    save_video = args.mode in {"video", "both"}
    save_frames = args.mode in {"frames", "both"}
    video_path, frames_dir, metadata_path = resolve_output_paths(args)
    depth_dir = resolve_depth_dir(args, frames_dir)
    save_depth = depth_dir is not None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if depth_dir is not None:
        depth_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    codec = choose_codec(video_path, args.codec) if video_path is not None else None
    depth_preview = args.show_depth and args.preview
    need_depth_stream = depth_preview or save_depth

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if need_depth_stream:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    print("Starting RealSense pipeline...")
    profile = pipeline.start(config)
    depth_scale = None
    if need_depth_stream:
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    metadata: dict[str, object] = build_metadata(
        profile,
        include_depth=need_depth_stream,
        depth_scale=depth_scale,
    )
    color_intrinsics = metadata["color_intrinsics"]
    frame_size = (
        int(color_intrinsics["width"]),
        int(color_intrinsics["height"]),
    )
    writer = None
    if save_video:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*codec),
            args.fps,
            frame_size,
        )
        if not writer.isOpened():
            pipeline.stop()
            raise SystemExit(
                f"Failed to open video writer for {video_path}. Try --codec XVID and an .avi output."
            )

    align = rs.align(rs.stream.color) if need_depth_stream else None
    preview_enabled = args.preview
    frame_count = 0
    clipped_depth_frames = 0
    last_frame_timestamp_ns = None
    started_at = datetime.now().isoformat(timespec="seconds")

    metadata.update(
        {
            "mode": args.mode,
            "output_mode": args.mode,
            "video_output_path": str(video_path) if video_path is not None else None,
            "frames_dir": str(frames_dir) if frames_dir is not None else None,
            "save_depth": save_depth,
            "depth_dir": str(depth_dir) if depth_dir is not None else None,
            "frame_filename_format": "YYYYMMDD_HHMMSS_mmm",
            "depth_png_scale": DEPTH_PNG_SCALE if save_depth else None,
            "metadata_path": str(metadata_path),
            "codec": codec,
            "image_format": FRAME_IMAGE_FORMAT if save_frames else None,
            "jpeg_quality": None,
            "requested_width": args.width,
            "requested_height": args.height,
            "requested_fps": args.fps,
            "actual_width": frame_size[0],
            "actual_height": frame_size[1],
            "warmup_frames": args.warmup_frames,
            "started_at": started_at,
        }
    )

    wall_start = time.time()

    try:
        for _ in range(max(args.warmup_frames, 0)):
            pipeline.wait_for_frames()

        if save_video:
            print(f"Recording video to {video_path}")
        if save_frames:
            print(f"Saving frames to {frames_dir}")
        if save_depth:
            print(f"Saving aligned depth to {depth_dir}")
        if preview_enabled:
            print("Press q in the preview window to stop recording.")
        else:
            print("Press Ctrl-C to stop recording.")

        wall_start = time.time()
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if need_depth_stream else None
            if not color_frame:
                continue
            if save_depth and not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            frame_timestamp_ns = next_frame_timestamp_ns(last_frame_timestamp_ns)
            frame_stem = format_frame_stem(frame_timestamp_ns)
            last_frame_timestamp_ns = frame_timestamp_ns
            if writer is not None:
                writer.write(color_image)
            if save_frames:
                save_frame(
                    image=color_image,
                    frame_stem=frame_stem,
                    frames_dir=frames_dir,
                )
            if save_depth:
                depth_meters = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(depth_scale)
                clipped_depth_frames += int(
                    save_depth_frame(
                        depth_meters=depth_meters,
                        frame_stem=frame_stem,
                        depth_dir=depth_dir,
                    )
                )
            frame_count += 1

            if preview_enabled:
                preview_image = color_image
                if depth_preview:
                    if depth_frame:
                        depth_image = np.asanyarray(depth_frame.get_data())
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_image, alpha=0.03),
                            cv2.COLORMAP_JET,
                        )
                        preview_image = cv2.hconcat([color_image, depth_colormap])
                try:
                    cv2.imshow("RealSense D435i Recorder", preview_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    preview_enabled = False
                    print("OpenCV preview is not available in this environment. Continuing without preview.")

            if args.duration > 0 and (time.time() - wall_start) >= args.duration:
                break

        elapsed = max(time.time() - wall_start, 1e-6)
        metadata.update(
            {
                "stopped_at": datetime.now().isoformat(timespec="seconds"),
                "duration_sec": elapsed,
                "frame_count": frame_count,
                "average_fps": frame_count / elapsed,
                "depth_frames_clipped_to_uint16": clipped_depth_frames if save_depth else 0,
            }
        )
    except KeyboardInterrupt:
        elapsed = max(time.time() - wall_start, 1e-6)
        metadata.update(
            {
                "stopped_at": datetime.now().isoformat(timespec="seconds"),
                "duration_sec": elapsed,
                "frame_count": frame_count,
                "average_fps": frame_count / elapsed,
                "depth_frames_clipped_to_uint16": clipped_depth_frames if save_depth else 0,
            }
        )
        print("\nRecording interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if save_video:
        print(f"Saved video: {video_path}")
    if save_frames:
        print(f"Saved frames: {frames_dir}")
    if save_depth:
        print(f"Saved depth: {depth_dir}")
    print(f"Saved metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

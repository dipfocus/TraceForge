import os
import numpy as np
import cv2
import mediapy as media
import torch
from PIL import Image
import math
import tqdm
import glob
from rich import print
import argparse
from loguru import logger
import json

from utils.video_depth_pose_utils import video_depth_pose_dict

from datasets.data_ops import _filter_one_depth
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from utils.inference_utils import load_model, inference
from utils.threed_utils import (
    project_tracks_3d_to_2d,
    project_tracks_3d_to_3d,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to video directory (for batch processing) or single video folder",
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        default=None,
        help="Path to depth directory (if known depth is provided) for batch processing or single video folder",
    )
    parser.add_argument(
        "--depth_png_scale",
        type=float,
        default=10000.0,
        help="Decode 16-bit depth PNGs from --depth_path as meters via value / depth_png_scale.",
    )
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/tapip3d_final.pth"
    )
    parser.add_argument('--depth_pose_method', type=str, default='vggt4', choices=video_depth_pose_dict.keys())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_iters", type=int, default=6)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--max_num_frames", type=int, default=384)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Trajectory horizon length for each sample",
    )
    parser.add_argument(
        "--batch_process",
        action="store_true",
        default=False,
        help="Process all video folders in the given directory",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip processing if output already exists",
    )
    parser.add_argument(
        "--use_all_trajectories",
        action="store_true",
        default=True,
        help="Include all visible trajectories in each frame (default: True)",
    )
    parser.add_argument(
        "--frame_drop_rate",
        type=int,
        default=1,
        help="Query uniform grid points every N frames (default: 1, query every frame)",
    )
    parser.add_argument(
        "--scan_depth",
        type=int,
        default=2,  # default depth changed to 2
        help="How many directory levels below --video_path to scan for subfolders "
            "when --batch_process is enabled. Default is 2 (e.g., P02_02_01)."
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=128,
        help="Tracking window length (number of frames) per query frame in offline mode",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=50,
        help="Target max frames to keep per episode. If --fps <= 0, use stride = ceil(N / max_frames_per_video).",
    )   
    return parser.parse_args()

def retarget_trajectories(
    trajectory: np.ndarray,
    interval: float = 0.05,
    max_length: int = 64,
    top_percent: float = 0.02,
):
    """
    Synchronous arc-length retargeting using per-segment robust speeds.

    Steps:
      1) Global normalize x,y by (trajectory[-1,0,0], trajectory[-1,0,1]), then clip x,y to [0,1].
      2) For each time segment t: compute lengths for all tracks; take mean of top `top_percent`
         → robust_seglen[t].
      3) Build cumulative arc-length from robust_seglen and place targets every `interval`.
         (Long segments get subdivided; short ones merge implicitly.)
      4) For each target in segment t with fraction alpha, interpolate *all* tracks
         between frames t and t+1 with the same alpha (synchronous).
      5) Denormalize x,y only; z (if present) is linearly interpolated without scaling.

    Args:
        trajectory: (N, H, D) with D in {2,3}
        interval: target arc-length step
        max_length: output max length
        top_percent: fraction (0,1] for robust top-k mean per segment (e.g., 0.02 = top 2%)

    Returns:
        retargeted: (N, max_length, D), padded with -np.inf
        valid_mask: (max_length) bool
    """
    assert trajectory.ndim == 3, "trajectory must be (N, H, D)"
    N, H, D = trajectory.shape
    assert D in (2, 3), "D must be 2 or 3"
    if not (0 < top_percent <= 1.0):
        raise ValueError("top_percent must be in (0, 1].")
    if interval <= 0:
        raise ValueError("interval must be > 0")
    if H < 2:
        # If H==1, there is no segment to interpolate → return only the first frame
        ret = np.full((N, max_length, D), -np.inf, dtype=trajectory.dtype)
        mask = np.zeros((max_length), dtype=bool)
        ret[:, 0, :] = trajectory[:, 0, :]
        mask[0] = True
        return ret, mask

    eps = 1e-12

    # ---- 1) Global normalization (x,y) & clipping ----
    scale_x = float(trajectory[-1, 0, 0])
    scale_y = float(trajectory[-1, 0, 1])
    if abs(scale_x) < eps: scale_x = 1.0
    if abs(scale_y) < eps: scale_y = 1.0

    traj_norm = trajectory.astype(np.float64, copy=True)
    traj_norm[:, :, 0] /= scale_x
    traj_norm[:, :, 1] /= scale_y
    # clip x,y to [0,1]
    np.clip(traj_norm[:, :, 0], 0.0, 1.0, out=traj_norm[:, :, 0])
    np.clip(traj_norm[:, :, 1], 0.0, 1.0, out=traj_norm[:, :, 1])
    # z is not scaled/clipped

    # ---- 2) Robust length per segment t: mean of top k% ----
    # seglens_all: (N, H-1)
    diffs_all = traj_norm[:, 1:, :] - traj_norm[:, :-1, :]
    seglens_all = np.linalg.norm(diffs_all, axis=2)

    k = max(1, int(np.ceil(top_percent * N)))
    # Use np.partition to get per-segment (column-wise) top-k without full sorting
    # Values below index N-k are smaller; values at/above are larger
    part = np.partition(seglens_all, N - k, axis=0)      # (N, H-1)
    topk = part[N - k:, :]                                # (k, H-1)
    robust_seglen = topk.mean(axis=0)                     # (H-1,)

    total_len = float(robust_seglen.sum())
    # Output buffers
    retargeted = np.full((N, max_length, D), -np.inf, dtype=trajectory.dtype)
    valid_mask = np.zeros((max_length), dtype=bool)

    # ---- 3) Create targets at 'interval' along the robust cumulative length ----
    k_max = int(np.floor(total_len / interval))
    num_samples = min(k_max + 1, max_length)
    targets = interval * np.arange(num_samples, dtype=np.float64)
    targets[-1] = min(targets[-1], total_len)

    # Cumulative length s (vertex-based): s[0]=0, s[i]=sum_{j<i} robust_seglen[j]
    s = np.zeros((H,), dtype=np.float64)
    s[1:] = np.cumsum(robust_seglen, dtype=np.float64)

    # Segment index and in-segment fraction alpha for each target
    idx_seq = np.searchsorted(s, targets, side='right') - 1   # (num_samples,)
    idx_seq = np.clip(idx_seq, 0, H - 2)
    denom = np.maximum(robust_seglen[idx_seq], eps)           # (num_samples,)
    alpha = (targets - s[idx_seq]) / denom                    # (num_samples,)
    alpha_seq = alpha.reshape(-1, 1)                          # (num_samples,1)

    # ---- 4) Synchronous interpolation: apply the same (idx, alpha) to all tracks ----
    left = traj_norm[:, idx_seq, :]           # (N, num_samples, D)
    right = traj_norm[:, idx_seq + 1, :]      # (N, num_samples, D)
    samples_norm = left + alpha_seq[None, :, :] * (right - left)  # (N, num_samples, D)

    # ---- 5) Denormalize: scale only x,y back ----
    samples_out = samples_norm.astype(trajectory.dtype, copy=True)
    samples_out[:, :, 0] *= scale_x
    samples_out[:, :, 1] *= scale_y
    # Keep z as the linear interpolation result

    L = num_samples
    retargeted[:, :L, :] = samples_out
    valid_mask[:L] = True
    return retargeted, valid_mask

def save_structured_data(
    video_name,
    output_dir,
    video_tensor,
    depths,
    coords,
    visibs,
    intrinsics,
    extrinsics,
    query_points_per_frame,
    horizon,
    original_filenames,
    use_all_trajectories=True,
    query_frame_results=None,
    future_len: int = 128,
):
    """Save data in the structured format"""

    # Create output directories
    video_output_dir = os.path.join(output_dir, video_name)
    images_dir = os.path.join(video_output_dir, "images")
    depth_dir = os.path.join(video_output_dir, "depth")
    samples_dir = os.path.join(video_output_dir, "samples")

    # Save structured data in the new format
    for dir_path in [images_dir, depth_dir, samples_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # If we have query_frame_results, save each query frame's results independently
    if query_frame_results is not None:
        logger.info(f"Processing {len(query_frame_results)} query frame results")

        saved_count = 0

        for query_frame_idx, frame_data in query_frame_results.items():
            coords_np = frame_data["coords"].cpu().numpy()  # (T, 400, 3)

            # Save RGB images for this segment
            video_segment = frame_data["video_segment"].cpu().numpy() * 255
            video_segment = video_segment.astype(np.uint8).transpose(
                0, 2, 3, 1
            )  # (T, H, W, 3)


            # Save sample data for this query frame
            coords_np = frame_data["coords"].cpu().numpy()  # (T, 400, 3) where T <= 16
            visibs_np = frame_data["visibs"].cpu().numpy()  # (T, 400)
            intrinsics_np = frame_data["intrinsics_segment"].cpu().numpy()  # (T, 3, 3)
            extrinsics_np = frame_data["extrinsics_segment"].cpu().numpy()  # (T, 4, 4)

            # Debug: Check shapes
            logger.debug(
                f"Query frame {query_frame_idx}: coords_np shape = {coords_np.shape}"
            )
            logger.debug(
                f"Query frame {query_frame_idx}: visibs_np shape = {visibs_np.shape}"
            )

            # Handle edge cases where coords might not have expected dimensions
            if len(coords_np.shape) != 3:
                logger.error(
                    f"Unexpected coords shape for frame {query_frame_idx}: {coords_np.shape}"
                )
                continue

            # Get actual number of frames in this segment
            actual_frames = coords_np.shape[0]

            # Create sample data for this query frame
            sample_data = {}

            # Grid points (20x20 = 400 points) for this query frame
            grid_size = 20
            frame_h, frame_w = video_segment.shape[1:3]
            y_coords = np.linspace(0, frame_h - 1, grid_size)
            x_coords = np.linspace(0, frame_w - 1, grid_size)
            xx, yy = np.meshgrid(x_coords, y_coords)
            keypoints = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (400, 2)

            sample_data["image_path"] = np.array(
                [f"images/{video_name}_{query_frame_idx}.png"], dtype="<U50"
            )
            sample_data["frame_index"] = np.array([query_frame_idx])
            sample_data["keypoints"] = keypoints.astype(np.float32)  # (400, 2)

            # trajectories: (400, T, 3) - 400 tracks, T frames (T <= 16), xyz coordinates
            try:
                sample_data["traj"] = coords_np.transpose(1, 0, 2).astype(
                    np.float32
                )  # (400, T, 3)
            except ValueError as e:
                logger.error(
                    f"Error transposing coords for frame {query_frame_idx}: {e}"
                )
                logger.error(f"coords_np shape: {coords_np.shape}")
                # Skip this frame and continue
                continue

            # Project 3D coordinates to 2D for traj_2d
            camera_views_segment = []
            for t in range(len(intrinsics_np)):
                camera_views_segment.append(
                    {
                        "c2w": np.linalg.inv(extrinsics_np[t]),
                        "K": intrinsics_np[t],
                        "height": frame_h,
                        "width": frame_w,
                    }
                )

            # Use the first frame's camera for consistent projection
            fixed_camera_view = camera_views_segment[0]

            # Project to 2D using the same camera view
            coords_3d_for_projection = coords_np  # (16, 400, 3)
            try:
                tracks2d_fixed = project_tracks_3d_to_2d(
                    tracks3d=coords_3d_for_projection,
                    camera_views=[fixed_camera_view] * len(coords_3d_for_projection),
                )  # (T, 400, 2)
                tracks3d_fixed = project_tracks_3d_to_3d(
                    tracks3d=coords_3d_for_projection,
                    camera_views=[fixed_camera_view] * len(coords_3d_for_projection),
                )  # (T, 400, 3)

                sample_data["traj_2d"] = tracks2d_fixed.transpose(1, 0, 2).astype(
                    np.float32
                )  # (400, T, 2)
                sample_data["traj"] = tracks3d_fixed.transpose(1, 0, 2).astype(
                    np.float32
                )  # (400, T, 3)
            except Exception as e:
                logger.error(
                    f"Error projecting tracks for frame {query_frame_idx}: {e}"
                )
                # Fallback: use original coordinates
                sample_data["traj_2d"] = (
                    coords_np[:, :, :2].transpose(1, 0, 2).astype(np.float32)
                )
                sample_data["traj"] = coords_np.transpose(1, 0, 2).astype(np.float32)

            # Only save image and depth for the query frame itself, not the entire segment
            query_frame_img = video_segment[
                0
            ]  # First frame in segment is the query frame
            query_frame_depth = (
                frame_data["depths_segment"].cpu().numpy()[0]
            )  # First depth

            img_filename = f"{video_name}_{query_frame_idx}.png"
            img_path = os.path.join(images_dir, img_filename)
            if not os.path.exists(img_path):  # Avoid duplicate saves
                Image.fromarray(query_frame_img).save(img_path)

            # Save depth image for query frame only
            depth_filename = f"{video_name}_{query_frame_idx}.png"
            depth_path = os.path.join(depth_dir, depth_filename)
            if not os.path.exists(depth_path):  # Avoid duplicate saves
                # Normalize depth for visualization and save as 16-bit PNG
                depth_normalized = (query_frame_depth * 10000).astype(np.uint16)
                Image.fromarray(depth_normalized, mode="I;16").save(depth_path)

                # save depth raw value as npz
                depth_raw_filename = f"{video_name}_{query_frame_idx}_raw.npz"
                depth_raw_path = os.path.join(depth_dir, depth_raw_filename)
                np.savez(depth_raw_path, depth=query_frame_depth)
            
            retargeted, valid_mask = retarget_trajectories(sample_data["traj"], max_length=args.future_len)
            sample_data["traj"] = retargeted
            sample_data["valid_steps"] = valid_mask

            # Save sample NPZ for this query frame
            sample_filename = f"{video_name}_{query_frame_idx}.npz"
            sample_path = os.path.join(samples_dir, sample_filename)
            np.savez(sample_path, **sample_data)

            logger.info(
                f"Saved query frame {query_frame_idx} with 400 trajectories tracked for {actual_frames} frames"
            )
            saved_count += 1

        logger.info(f"Saved {saved_count} frames")


def process_single_video(video_path, depth_path, args, model_3dtracker, model_depth_pose):
    """Process a single video and return the processed data"""
    logger.info(f"Processing video: {video_path}")

    # --- NEW: per-episode stride based on frame count when --fps <= 0 ---
    # If user set --fps > 0, use that fixed stride; otherwise auto-compute from N.
    if args.fps and int(args.fps) > 0:
        stride = int(args.fps)
        n_frames = 0  # unknown/not needed in fixed stride mode
    else:
        stride = 1
        n_frames = 0
        if os.path.isdir(video_path):
            # Count frames by scanning image files in the episode folder
            img_files = []
            for ext in ["jpg", "jpeg", "png"]:
                img_files.extend(glob.glob(os.path.join(video_path, f"*.{ext}")))
            n_frames = len(img_files)

            # Auto stride: ceil(N / target), where target = --max_frames_per_video
            target = max(1, int(getattr(args, "max_frames_per_video", 150)))
            stride = max(1, math.ceil(n_frames / target)) if n_frames > 0 else 1
        else:
            # For video files (.mp4, etc.), we keep stride=1 (or you can extend to probe length)
            stride = 1

    logger.info(
        f"[{os.path.basename(video_path)}] frames={n_frames if n_frames else 'n/a'} "
        f"target={getattr(args, 'max_frames_per_video', 150)} -> stride={stride}"
    )

    # Load RGB with computed stride
    video_tensor, video_mask, original_filenames = load_video_and_mask(
        video_path,
        args.mask_dir,
        stride,
        args.max_num_frames,
        depth_png_scale=args.depth_png_scale,
    )

    # Load depth (if provided) with the SAME stride to keep alignment with RGB
    depth_tensor = None
    if depth_path is not None:
        depth_tensor, _, _ = load_video_and_mask(
            depth_path,
            None,
            stride,
            args.max_num_frames,
            is_depth=True,
            depth_png_scale=args.depth_png_scale,
        )  # [T, H, W]
        valid_depth = (depth_tensor > 0)
        depth_tensor[~valid_depth] = 0  # Invalidate bad depth values

    video_length = len(video_tensor)

    # obtain video depth and pose
    (
        video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy
    ) = model_depth_pose(
        video_tensor,
        known_depth=depth_tensor,  # can be None
        stationary_camera=False,
        replace_with_known_depth=False,  # align scale to known depth but keep model-predicted depth map
    )

    # Keep depth_conf for visualization NPZ
    depth_conf_npy = depth_conf.squeeze().cpu().numpy()

    frame_H, frame_W = video_ten.shape[-2:]

    # Sample query points using uniform grid and store which frame they belong to
    query_points_per_frame = {}

    # Use uniform grid sampling (20x20 = 400 points per frame)
    query_point = []
    tracking_segments = []  # Store info about which frames to track for each segment

    # Determine which frames to query based on frame_drop_rate
    query_frames = list(range(0, video_length, args.frame_drop_rate))
    logger.info(
        f"Using uniform grid sampling on frames: {query_frames} (frame_drop_rate={args.frame_drop_rate})"
    )
    logger.info(f"Tracking up to {args.future_len} frames from each query frame")

    for frame_idx in query_frames:
        # Calculate the end frame for this tracking segment (16 frames max)
        end_frame = min(frame_idx + args.future_len, video_length)
        tracking_segments.append((frame_idx, end_frame))

        # Create 20x20 uniform grid for this frame
        grid_points = (
            create_uniform_grid_points(
                height=frame_H, width=frame_W, grid_size=20, device="cpu"
            )
            .squeeze(0)
            .numpy()
        )  # Remove batch dimension and convert to numpy

        # Set the correct frame index for all points
        grid_points[:, 0] = frame_idx

        query_point.append(grid_points)

    # Group query points by frame
    for query_frame_points in query_point:
        if len(query_frame_points) > 0:
            frame_idx = int(query_frame_points[0, 0])
            points_xy = query_frame_points[:, 1:3]  # Extract x, y coordinates
            query_points_per_frame[frame_idx] = points_xy

    # Process each query frame independently with 16-frame tracking
    extrs_npy = np.linalg.inv(extrs_npy)

    # Store results for each query frame
    query_frame_results = {}

    logger.info(f"Processing {len(tracking_segments)} independent tracking segments")

    for seg_idx, (start_frame, end_frame) in enumerate(tracking_segments):
        logger.info(
            f"Processing query frame {start_frame}: tracking {end_frame - start_frame} frames"
        )

        # Clear CUDA cache before each segment to avoid fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract video segment (16 frames starting from query frame)
        video_segment = video_ten[start_frame:end_frame]
        depth_segment = depth_npy[start_frame:end_frame]
        intrs_segment = intrs_npy[start_frame:end_frame]
        extrs_segment = extrs_npy[start_frame:end_frame]

        # Get query points for this segment (only from the starting frame)
        # Need to adjust the frame index to be relative to segment start (0)
        segment_query_point = [query_point[seg_idx].copy()]
        segment_query_point[0][:, 0] = 0  # Set frame index to 0 for segment start

        video, depths, intrinsics, extrinsics, query_point_tensor, support_grid_size = (
            prepare_inputs(
                video_segment,
                depth_segment,
                intrs_segment,
                extrs_segment,
                segment_query_point,
                inference_res=(frame_H, frame_W),
                support_grid_size=16,
                device=args.device,
            )
        )

        model_3dtracker.set_image_size((frame_H, frame_W))

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                coords_seg, visibs_seg = inference(
                    model=model_3dtracker,
                    video=video,
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    query_point=query_point_tensor,
                    num_iters=args.num_iters,
                    grid_size=support_grid_size,
                    bidrectional=False,  # Disable backward tracking
                )

        # Validate inference results before storing
        logger.debug(
            f"Query frame {start_frame}: coords_seg shape = {coords_seg.shape}, visibs_seg shape = {visibs_seg.shape}"
        )

        # Check if results have expected dimensions
        if len(coords_seg.shape) != 3 or len(visibs_seg.shape) != 2:
            logger.error(
                f"Query frame {start_frame}: Invalid result shapes - coords: {coords_seg.shape}, visibs: {visibs_seg.shape}"
            )
            continue

        # Check if we have the expected number of trajectories (400)
        expected_trajectories = 400
        if coords_seg.shape[1] != expected_trajectories:
            logger.warning(
                f"Query frame {start_frame}: Expected {expected_trajectories} trajectories, got {coords_seg.shape[1]}"
            )

        # Store results for this query frame
        query_frame_results[start_frame] = {
            "coords": coords_seg,  # Shape: (T, 400, 3)
            "visibs": visibs_seg,  # Shape: (T, 400)
            "video_segment": video,
            "depths_segment": depths,
            "intrinsics_segment": intrinsics,
            "extrinsics_segment": extrinsics,
        }

        logger.info(
            f"Query frame {start_frame}: tracked {coords_seg.shape[1]} trajectories for {coords_seg.shape[0]} frames"
        )
        
        # Clear cache after inference to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # For compatibility with the rest of the pipeline, use the first segment as the main result
    # But we'll save each segment independently in save_structured_data
    if query_frame_results:
        first_frame = min(query_frame_results.keys())
        coords = query_frame_results[first_frame]["coords"]
        visibs = query_frame_results[first_frame]["visibs"]
        video = query_frame_results[first_frame]["video_segment"]
        depths = query_frame_results[first_frame]["depths_segment"]
        intrinsics = query_frame_results[first_frame]["intrinsics_segment"]
        extrinsics = query_frame_results[first_frame]["extrinsics_segment"]
    else:
        flen = min(args.future_len, len(video_ten))
        coords = torch.empty((0, 0, 3))
        visibs = torch.empty((0, 0))
        video = video_ten[:flen]
        depths = torch.from_numpy(depth_npy[:flen]).float().to(args.device)
        intrinsics = torch.from_numpy(intrs_npy[:flen]).float().to(args.device)
        extrinsics = torch.from_numpy(extrs_npy[:flen]).float().to(args.device)

    # Validate tensor shapes after inference
    logger.debug(
        f"After inference - coords shape: {coords.shape}, visibs shape: {visibs.shape}"
    )

    # Ensure visibs has the expected dimensions
    if visibs.dim() == 3 and visibs.shape[-1] == 1:
        visibs = visibs.squeeze(-1)  # Remove last dimension if it's 1
        logger.debug(f"Squeezed visibs shape: {visibs.shape}")

    # Validate final shapes
    expected_frames = video.shape[0]
    expected_points = coords.shape[1] if coords.dim() >= 2 else 0
    if coords.dim() != 3 or visibs.dim() != 2:
        logger.error(
            f"Unexpected tensor dimensions - coords: {coords.shape}, visibs: {visibs.shape}"
        )
        raise ValueError(f"Invalid tensor shapes after inference")

    if coords.shape[0] != expected_frames or visibs.shape[0] != expected_frames:
        logger.error(
            f"Frame count mismatch - expected {expected_frames}, got coords: {coords.shape[0]}, visibs: {visibs.shape[0]}"
        )
        raise ValueError(f"Frame count mismatch in inference results")

    return {
        "video_tensor": video,
        "depths": depths,
        "coords": coords,
        "visibs": visibs,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "query_points_per_frame": query_points_per_frame,
        "original_filenames": original_filenames,
        "depth_conf": depth_conf_npy,
        "query_frame_results": query_frame_results,  # Add individual frame results
        "full_intrinsics": torch.from_numpy(intrs_npy)
        .float()
        .to(args.device),  # Full video intrinsics
        "full_extrinsics": torch.from_numpy(extrs_npy)
        .float()
        .to(args.device),  # Full video extrinsics
    }


def find_video_folders(base_path: str, scan_depth: int = 2):
    """
    Recursively scan subfolders up to a given depth and return inputs
    that contain images (.jpg/.jpeg/.png) or stand-alone video files
    (.mp4/.webm/etc.).

    Args:
        base_path: Root directory to scan
        scan_depth: Number of directory levels to traverse

    Returns:
        List of folder paths containing image files at the target depth
    """
    img_exts = (".jpg", ".jpeg", ".png")
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg")

    # Normalize the base path
    base_path = os.path.abspath(base_path.rstrip(os.sep))
    base_depth = base_path.count(os.sep)
    target_depth = base_depth + scan_depth

    video_folders = []

    for root, dirs, files in os.walk(base_path):
        current_depth = os.path.abspath(root.rstrip(os.sep)).count(os.sep)

        # Skip folders above the target depth
        if current_depth < target_depth:
            continue

        # Select only folders/files exactly at the target depth
        if current_depth == target_depth:
            has_images = any(f.lower().endswith(img_exts) for f in files)
            if has_images:
                video_folders.append(root)
            # Also collect individual video files at this depth
            for f in files:
                if f.lower().endswith(video_exts):
                    video_folders.append(os.path.join(root, f))

        # Skip deeper folders for performance (no need to go further)
        if current_depth > target_depth:
            dirs[:] = []  # prevent os.walk from descending further

    # Deduplicate and sort for stable ordering
    video_folders = sorted(list(dict.fromkeys(video_folders)))
    return video_folders


def load_depth_file(depth_path, depth_png_scale=10000.0):
    raw_npz_path = os.path.splitext(depth_path)[0] + "_raw.npz"
    if os.path.exists(raw_npz_path):
        with np.load(raw_npz_path) as depth_data:
            if "depth" in depth_data:
                return depth_data["depth"].astype(np.float32)
            return depth_data[depth_data.files[0]].astype(np.float32)

    depth_img = Image.open(depth_path).convert("I;16")
    depth = np.array(depth_img).astype(np.float32)
    if depth_png_scale > 0:
        depth /= float(depth_png_scale)
    return depth


def load_video_and_mask(
    video_path,
    mask_dir=None,
    fps=1,
    max_num_frames=384,
    is_depth=False,
    depth_png_scale=10000.0,
):
    original_filenames = []

    if os.path.isdir(video_path):
        img_files = []
        for ext in ["jpg", "png"]:
            img_files.extend(sorted(glob.glob(os.path.join(video_path, f"*.{ext}"))))

        # IMPORTANT: Subsample the file list BEFORE loading to save memory
        img_files = img_files[::fps]

        video_tensor = []
        for img_file in tqdm.tqdm(img_files, desc="Loading images"):
            if is_depth:
                video_tensor.append(
                    torch.from_numpy(load_depth_file(img_file, depth_png_scale)).float()
                )
            else:
                img = Image.open(img_file)
                img = img.convert("RGB")
                video_tensor.append(
                    torch.from_numpy(np.array(img)).float()
                )
            # Extract original filename without extension
            filename = os.path.splitext(os.path.basename(img_file))[0]
            original_filenames.append(filename)
        video_tensor = torch.stack(video_tensor)  # (N, H, W, 3)
    elif video_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        # simple video reading. Please modify it if it causes OOM
        video_tensor = torch.from_numpy(media.read_video(video_path))
        # Generate frame names for video files
        for i in range(len(video_tensor)):
            original_filenames.append(f"frame_{i:010d}")
        # For video files, subsample after loading
        video_tensor = video_tensor[::fps]
        original_filenames = original_filenames[::fps]

    if not is_depth:
        video_tensor = video_tensor.permute(
            0, 3, 1, 2
        )  # Convert to tensor and permute to (N, C, H, W)
    video_tensor = video_tensor.float()
    video_tensor = video_tensor[:max_num_frames]
    original_filenames = original_filenames[:max_num_frames]
    video_length = len(video_tensor)
    logger.debug(f"Loaded video with {video_length} frames from {video_path}")
    frame_h, frame_w = video_tensor.shape[-2:]

    video_mask_npy = None
    if mask_dir is not None:
        video_mask_npy = []
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        for mask_file in mask_files:
            mask = media.read_image(mask_file)
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            video_mask_npy.append(mask)
        video_mask_npy = np.stack(video_mask_npy)

    if not is_depth:
        video_tensor /= 255.
    return video_tensor, video_mask_npy, original_filenames


def create_uniform_grid_points(height, width, grid_size=20, device="cuda"):
    """Create uniform grid points across the image.

    Args:
        height (int): Image height
        width (int): Image width
        grid_size (int): Grid size (20x20)
        device (str): Device for tensor

    Returns:
        torch.Tensor: Grid points [1, grid_size*grid_size, 3] where each point is [t, x, y]
    """
    # Create uniform grid
    y_coords = np.linspace(0, height - 1, grid_size)
    x_coords = np.linspace(0, width - 1, grid_size)

    # Create meshgrid
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Flatten and create points [N, 2]
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)

    # Add time dimension (t=0 for all points) -> [N, 3]
    time_col = np.zeros((grid_points.shape[0], 1))
    grid_points_3d = np.concatenate([time_col, grid_points], axis=1)

    # Convert to tensor and add batch dimension -> [1, N, 3]
    grid_tensor = torch.tensor(
        grid_points_3d, dtype=torch.float32, device=device
    ).unsqueeze(0)

    return grid_tensor

def prepare_query_points(query_xyt, depths, intrinsics, extrinsics):
    final_queries = []
    for query_i in query_xyt:
        if len(query_i) == 0:
            continue

        t = int(query_i[0, 0])
        depth_t = depths[t]
        K_inv_t = np.linalg.inv(intrinsics[t])
        c2w_t = np.linalg.inv(extrinsics[t])

        xy = query_i[:, 1:]
        ji = np.round(xy).astype(int)
        d = depth_t[ji[..., 1], ji[..., 0]]
        xy_homo = np.concatenate([xy, np.ones_like(xy[:, :1])], axis=-1)
        local_coords = K_inv_t @ xy_homo.T  # (3, N)
        local_coords = local_coords * d[None, :]  # (3, N)
        world_coords = c2w_t[:3, :3] @ local_coords + c2w_t[:3, 3:]
        final_queries.append(np.concatenate([query_i[:, :1], world_coords.T], axis=-1))
    return np.concatenate(final_queries, axis=0)  # (N, 4)


def prepare_inputs(
    video_ten,
    depths,
    intrinsics,
    extrinsics,
    query_point,
    inference_res: Tuple[int, int],
    support_grid_size: int,
    num_threads: int = 8,
    device: str = "cuda",
):
    _original_res = depths.shape[1:3]
    inference_res = _original_res  # fix as the same

    intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)

    # resize & remove edges
    with ThreadPoolExecutor(num_threads) as executor:
        depths_futures = [
            executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic)
            for depth, intrinsic in zip(depths, intrinsics)
        ]
        depths = np.stack([future.result() for future in depths_futures])

    query_point = prepare_query_points(query_point, depths, intrinsics, extrinsics)
    query_point = torch.from_numpy(query_point).float().to(device)
    video = (video_ten.float()).to(device).clamp(0, 1)
    depths = torch.from_numpy(depths).float().to(device)
    intrinsics = torch.from_numpy(intrinsics).float().to(device)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)

    return video, depths, intrinsics, extrinsics, query_point, support_grid_size


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # initialize 3D models
    model_depth_pose = video_depth_pose_dict[args.depth_pose_method](args)
    model_3dtracker = load_model(args.checkpoint).to(args.device)

    # Determine video paths to process
    if args.batch_process:
        video_folders = find_video_folders(args.video_path, args.scan_depth)
        if args.depth_path is not None:
            depth_folders = find_video_folders(args.depth_path)
            if len(depth_folders) != len(video_folders):
                logger.error(
                    f"Number of depth folders ({len(depth_folders)}) does not match number of video folders ({len(video_folders)})"
                )
                exit(1)
        else:
            depth_folders = [None] * len(video_folders)

        logger.info(f"Found {len(video_folders)} video folders to process")
        if not video_folders:
            logger.error(f"No video folders found in {args.video_path}")
            exit(1)
    else:
        video_folders = [args.video_path]
        depth_folders = [args.depth_path]

    # Process each video
    for video_path, depth_path in zip(video_folders, depth_folders):
        video_name = os.path.basename(video_path.rstrip("/"))

        # Check if output already exists and skip if requested
        if args.skip_existing:
            output_path = os.path.join(out_dir, video_name)
            if os.path.exists(output_path):
                logger.info(f"Skipping {video_name} - output already exists")
                continue

        try:
            # Clear CUDA cache before processing each video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process video
            result = process_single_video(video_path, depth_path, args, model_3dtracker, model_depth_pose)

            # Save structured data
            save_structured_data(
                video_name=video_name,
                output_dir=out_dir,
                video_tensor=result["video_tensor"],
                depths=result["depths"],
                coords=result["coords"],
                visibs=result["visibs"],
                intrinsics=result["intrinsics"],
                extrinsics=result["extrinsics"],
                query_points_per_frame=result["query_points_per_frame"],
                horizon=args.horizon,
                original_filenames=result["original_filenames"],
                use_all_trajectories=args.use_all_trajectories,
                query_frame_results=result.get("query_frame_results"),
                future_len=args.future_len,
            )

            # Always save traditional visualization NPZ in video directory root
            video_dir = os.path.join(out_dir, video_name)
            data_npz_load = {}
            data_npz_load["coords"] = result["coords"].cpu().numpy()
            # Use full video camera parameters instead of segmented ones
            data_npz_load["extrinsics"] = result["full_extrinsics"].cpu().numpy()
            data_npz_load["intrinsics"] = result["full_intrinsics"].cpu().numpy()
            data_npz_load["height"] = result["video_tensor"].shape[-2]
            data_npz_load["width"] = result["video_tensor"].shape[-1]
            data_npz_load["depths"] = result["depths"].cpu().numpy().astype(np.float16)
            data_npz_load["unc_metric"] = result["depth_conf"].astype(np.float16)
            data_npz_load["visibs"] = result["visibs"][..., None].cpu().numpy()
            if args.save_video:
                data_npz_load["video"] = result["video_tensor"].cpu().numpy()

            save_path = os.path.join(video_dir, video_name + ".npz")
            np.savez(save_path, **data_npz_load)
            logger.info(f"Traditional visualization NPZ saved to {save_path}")

        except Exception as e:
            import traceback

            logger.error(f"Failed to process {video_name}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            continue

    # Cleanup
    del model_3dtracker
    del model_depth_pose
    torch.cuda.empty_cache()
    logger.info("Batch processing completed!")

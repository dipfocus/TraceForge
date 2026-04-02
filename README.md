# <img src="assets/trace_forge_logo.png" alt="TraceForge" height="25"> TraceForge

TraceForge is a unified dataset pipeline that converts cross-embodiment videos into consistent 3D traces via camera motion compensation and speed retargeting. 
For model training on the processed datasets, please refer to [TraceGen](https://github.com/jayLEE0301/TraceGen).

**Project Website**: [tracegen.github.io](https://tracegen.github.io/)  
**arXiv**: [2511.21690](https://arxiv.org/abs/2511.21690)

![TraceForge Overview](assets/TraceForge.png)

## Installation

### 1. Create a conda environment
```bash
conda create -n traceforge python=3.11
conda activate traceforge
```

### 2. Install dependencies 
Installs PyTorch 2.8.0 (CUDA 12.8) and all required packages.
```bash
bash setup_env.sh
```

### 3. Download checkpoints
Download the TAPIP3D model checkpoint:
```bash
mkdir -p checkpoints
wget -O checkpoints/tapip3d_final.pth https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
```

## Usage

### Prepare videos

**Record from an Intel RealSense D435i**
- `pyrealsense2` is included in `setup_env.sh`. If you did not use that script, install it first:
  ```bash
  pip install pyrealsense2
  ```
- Check the connected camera before recording:
  ```bash
  python tools/record_realsense.py --list-devices
  ```
- Print the full CLI help:
  ```bash
  python tools/record_realsense.py --help
  ```
- By default, the recorder saves image frames. If you do not pass output paths, it writes to `data/<timestamp>/`.
- To save frames under `data/` with an automatically generated timestamp subdirectory:
  ```bash
  python tools/record_realsense.py \
      --frames-dir data \
      --preview
  ```
- This creates a folder like `data/20260402_153000/` and writes `YYYYMMDD_HHMMSS_mmm.png` files there. Use the printed `Saved frames:` path as `infer.py --video_path`.
- If `--preview` is enabled, press `q` in the preview window to stop. Otherwise, stop with `Ctrl-C`.
- Record an `mp4` that can be passed directly to `infer.py`:
  ```bash
  python tools/record_realsense.py \
      --output-mode video \
      --output data/realsense/demo.mp4 \
      --preview \
      --show-depth
  ```
- The frame folder contains `YYYYMMDD_HHMMSS_mmm.png` files and can be used directly:
  ```bash
  python infer.py --video_path data/demo --out_dir outputs/demo
  ```
- To save aligned depth for `infer.py --depth_path` at the same time:
  ```bash
  python tools/record_realsense.py \
      --frames-dir data/realsense/demo \
      --save-depth \
      --preview \
      --show-depth
  ```
- Then run inference with both RGB and depth:
  ```bash
  python infer.py \
      --video_path data/realsense/demo \
      --depth_path data/realsense/demo/depth \
      --out_dir outputs/demo
  ```
- Record for a fixed amount of time instead of stopping manually:
  ```bash
  python tools/record_realsense.py \
      --frames-dir data/realsense/demo \
      --duration 10 \
      --preview
  ```
- Select one camera explicitly when multiple RealSense devices are connected:
  ```bash
  python tools/record_realsense.py \
      --serial <device_serial> \
      --frames-dir data/realsense/demo \
      --preview
  ```
- Use `--output-mode both` to save both the `mp4` and the frame folder in one run.
- Saved depth frames are aligned to RGB and encoded as 16-bit PNG with the same `/10000` convention used by TraceForge visualization outputs.
- The recorder also writes metadata with device info and intrinsics. When saving video, it is stored as a sidecar `.json`; when saving only frames, it is written to `recording_metadata.json` inside the frame folder.

**Common recorder arguments**

| Argument | Description | Default |
|----------|-------------|---------|
| `--output` | Base output path. Used to derive video path and/or frame directory when explicit paths are not provided. | `data/<timestamp>` |
| `--output-mode` | Save `video`, `frames`, or `both`. | `frames` |
| `--frames-dir` | Directory for RGB frames named like `YYYYMMDD_HHMMSS_mmm.png`. Passing `data` creates `data/<timestamp>/`. | Derived from `--output` |
| `--save-depth` | Save aligned depth PNGs to a `depth/` subdirectory under the resolved RGB frame directory, for example `data/<timestamp>/depth`. | `False` |
| `--duration` | Recording duration in seconds. Use `0` to record until stopped manually. | `0` |
| `--width` / `--height` / `--fps` | Requested stream resolution and FPS. | `1280 / 720 / 30` |
| `--serial` | Target device serial number when multiple cameras are connected. | `None` |
| `--codec` | Optional FOURCC override for video output, for example `mp4v` or `XVID`. | Auto by output suffix |
| `--warmup-frames` | Frames discarded before recording to let auto exposure settle. | `30` |
| `--preview` | Show an OpenCV preview window during recording. | `False` |
| `--show-depth` | Show aligned depth beside RGB in preview. Requires `--preview`. | `False` |

**Output notes**
- Frame folders produced by `tools/record_realsense.py` can be passed directly to `infer.py --video_path`.
- `--frames-dir data` is treated as a base directory, and the recorder creates `data/<timestamp>/` automatically.
- If `--output-mode video` is used and `--output` has no suffix, the recorder writes an `.mp4` file automatically.
- If `--save-depth` is enabled, the recorder creates a `depth/` subdirectory under the resolved RGB frame directory.
- With `--frames-dir data`, depth is written to `data/<timestamp>/depth/`.
- If OpenCV preview is unavailable in the current environment, recording continues without the preview window.

**Case A: videos directly in the input folder**
```
<input_video_directory>/
├── 1.webm
├── 2.webm
└── ...
```
- Use `--scan_depth 0` because the videos are already in the root folder.

**Case B: one subfolder per video containing extracted frames**
```
<input_video_directory>/
├── <video_name_1>/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── <video_name_2>/
│   ├── 000000.png
│   └── ...
└── ...
```
- Use `--scan_depth 1` so TraceForge scans one level down to reach each video’s frames.

**Case C: two-level layout (per-video folder with an `images/` subfolder)**
```
<input_video_directory>/
├── <video_name_1>/
│   └── images/
│       ├── 000000.png
│       ├── 000001.png
│       └── ...
├── <video_name_2>/
│   └── images/
│       ├── 000000.png
│       └── ...
└── ...
```
- Use `--scan_depth 2` to search two levels down for the image frames.

**Quick test dataset**
- Download a small sample dataset and unpack it under `data/test_dataset`:
  ```bash
  pip install gdown  # if not installed
  mkdir -p data
  gdown --fuzzy https://drive.google.com/file/d/1Vn1FNbthz-K8o2ijq9V7jYv10rElWuUd/view?usp=sharing -O data/test_dataset.tar
  tar -xf data/test_dataset.tar -C data
  ```
- The downloaded data follows the Case B layout above; run inference with 
    ```bash
    python infer.py \
        --video_path data/test_dataset \
        --out_dir <output_directory> \
        --batch_process \
        --use_all_trajectories \
        --skip_existing \
        --frame_drop_rate 5 \
        --scan_depth 1
    ```

### Running Inference
```bash
python infer.py \
    --video_path <input_video_directory> \
    --out_dir <output_directory> \
    --batch_process \
    --use_all_trajectories \
    --skip_existing \
    --frame_drop_rate 5 \
    --scan_depth 2
```

#### Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--video_path` | Path to video directory | Required |
| `--depth_path` | Path to aligned depth directory | `None` |
| `--depth_png_scale` | Decode `depth_path` PNG depth via `value / scale` | `10000` |
| `--out_dir` | Output directory | `outputs` |
| `--batch_process` | Process all video folders in the directory | `False` |
| `--skip_existing` | Skip if output already exists | `False` |
| `--frame_drop_rate` | Query points every N frames | `1` |
| `--scan_depth` | Directory levels to scan for subfolders | `2` |
| `--fps` | Frame sampling stride (0 for auto) | `1` |
| `--max_frames_per_video` | Target max frames per episode | `50` |
| `--future_len` | Tracking window length per query frame | `128` |

### Output Structure
```
<output_dir>/
└── <video_name>/
    ├── images/
    │   ├── <video_name>_0.png
    │   ├── <video_name>_5.png
    │   └── ...
    ├── depth/
    │   ├── <video_name>_0.png
    │   ├── <video_name>_0_raw.npz
    │   └── ...
    ├── samples/
    │   ├── <video_name>_0.npz
    │   ├── <video_name>_5.npz
    │   └── ...
    └── <video_name>.npz          # Full video visualization data
```

## Visualization

### 3D Trajectory Viewer
Visualize 3D traces on single images using viser:

<img src="assets/3dtrace_vis.png" alt="viser visualize" width="360">

```bash
python visualize_single_image.py \
    --npz_path <output_dir>/<video_name>/samples/<video_name>_0.npz \
    --image_path <output_dir>/<video_name>/images/<video_name>_0.png \
    --depth_path <output_dir>/<video_name>/depth/<video_name>_0.png \
    --port 8080
```

### Verify Output Files
Check saved NPZ files:
```bash
# 3D trajectory checker
python checker/batch_process_result_checker_3d.py <output_dir> --max-videos 1 --max-samples 3

# 2D trajectory checker
python checker/batch_process_result_checker.py <output_dir> --max-videos 1 --max-samples 3
```

## Instruction Generation

Generate task descriptions using VLM (Vision-Language Model).

### Setup API Keys
Create a `.env` file in the project root:
```bash
# For OpenAI (default)
OPENAI_API_KEY=your_openai_api_key

# For Google Gemini
GOOGLE_API_KEY=your_gemini_api_key
```

### Generate Descriptions
```bash
cd text_generation/
python generate_description.py --episode_dir <dataset_directory>

# Skip episodes that already have descriptions
python generate_description.py --episode_dir <dataset_directory> --skip_existing
```

## Helper Functions

- **Reading 3D data**: See `ThreedReader` in `visualize_single_image.py`
- **Point and camera transformations**: See `utils/threed_utils.py`

## 📖 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{lee2025tracegen,
  title={TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos},
  author={Lee, Seungjae and Jung, Yoonkyo and Chun, Inkook and Lee, Yao-Chih and Cai, Zikui and Huang, Hongjia and Talreja, Aayush and Dao, Tan Dat and Liang, Yongyuan and Huang, Jia-Bin and Huang, Furong},
  journal={arXiv preprint arXiv:2511.21690},
  year={2025}
}
```

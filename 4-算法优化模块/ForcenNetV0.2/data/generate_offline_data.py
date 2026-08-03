"""
Offline data generation script for ForCenNet.

Supports three modes:
1. UVDoc mode: Convert UVDoc_raw dataset to ForCenNet offline format.
2. Synthesize mode: Generate distorted pairs from clean document images using TPS.
3. UVDoc-final mode: Convert create_final.py output to ForCenNet offline format.

All processing (line detection, label generation) happens at native resolution.
Results are resized to --image_size only at the final save step.

Usage:
    # Mode 1: Convert from UVDoc_raw (default: save as 288x288)
    python data/generate_offline_data.py --mode uvdoc \
        --uvdoc_dir ../ArbDR/UVDoc_raw \
        --output_dir ./data/train

    # Mode 1: Save as 512x512
    python data/generate_offline_data.py --mode uvdoc \
        --uvdoc_dir ../ArbDR/UVDoc_raw \
        --output_dir ./data/train \
        --image_size 512

    # Mode 2: Synthesize from clean images
    python data/generate_offline_data.py --mode synthesize \
        --source_dir ./data/textures \
        --output_dir ./data/train \
        --num_per_image 50

    # Mode 3: Convert create_final.py output to offline format
    python data/generate_offline_data.py --mode uvdoc_final \
        --final_dir UVDoc_raw/final \
        --output_dir ./data/train_offline
"""

import os
import sys
import argparse
import glob
import json
import numpy as np
import cv2
import h5py
import hdf5storage as h5
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augmentation import (
    generate_random_distortion_field,
    warp_image,
    warp_mask,
    warp_points,
)
from data.label_generation import (
    generate_foreground_labels,
    sample_line_control_points,
)


def _save_line_points(output_dir: str, line_points_list: list, interval: int = 4):
    """
    Save line control points to disk as a single .npz file.

    Args:
        output_dir: Directory to save line_points.npz.
        line_points_list: List of (N_i, 2) arrays of line points in pixel coords.
        interval: Sampling interval for control points.
    """
    # Sample control points at regular intervals
    sampled = sample_line_control_points(line_points_list, interval=interval)

    if len(sampled) == 0 or all(len(pts) == 0 for pts in sampled):
        # No lines found — save empty placeholders
        np.savez(
            os.path.join(output_dir, "line_points.npz"),
            points=np.zeros((0, 2), dtype=np.float32),
            lengths=np.zeros(0, dtype=np.int64),
        )
        return

    # Concatenate all sampled points and record lengths
    lengths = []
    valid_arrays = []
    for pts in sampled:
        if len(pts) > 0:
            lengths.append(len(pts))
            valid_arrays.append(pts.astype(np.float32))

    all_points = np.concatenate(valid_arrays, axis=0)  # (N_total, 2)
    lengths_arr = np.array(lengths, dtype=np.int64)  # (num_lines,)

    np.savez(
        os.path.join(output_dir, "line_points.npz"),
        points=all_points,
        lengths=lengths_arr,
    )


def convert_uvdoc_sample(
    rgb_path: str,
    seg_path: str,
    uvmap_path: str,
    output_dir: str,
    image_size: int = 288,
    line_interval: int = 4,
):
    """
    Convert a single UVDoc_raw sample to ForCenNet offline format.

    All processing (line detection, label generation) happens at native resolution.
    Results are resized to image_size only at the final save step.

    Args:
        rgb_path: Path to distorted image (PNG).
        seg_path: Path to segmentation mask (.mat, HDF5).
        uvmap_path: Path to UV mapping field (.mat, HDF5).
        output_dir: Directory to save converted sample.
        image_size: Target image size (square) for saving output files.
        line_interval: Sampling interval for line control points.
    """
    os.makedirs(output_dir, exist_ok=True)

    native_size = (image_size, image_size) if image_size is not None else None

    # 1. Read distorted image at native resolution
    image = cv2.imread(rgb_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {rgb_path}")
    native_h, native_w = image.shape[:2]

    # 2. Read segmentation mask from HDF5 .mat at native resolution
    with h5py.File(seg_path, "r") as f:
        seg = np.array(f["seg"], dtype=np.uint8)  # (H, W)

    # UVDoc seg: 0=background, non-zero=foreground (document region)
    # Convert to binary mask: 1 = foreground, 0 = background
    mask = (seg > 0).astype(np.uint8) * 255
    # Resize mask to match image native resolution if needed
    if mask.shape[:2] != (native_h, native_w):
        mask = cv2.resize(mask, (native_w, native_h), interpolation=cv2.INTER_NEAREST)

    # 3. Read UV mapping field from HDF5 .mat at native resolution
    # NOTE: h5py reads HDF5 in C-order, which means MATLAB's (2, H, W) stored
    # in Fortran order is returned as (2, W, H) by h5py. We must use (2, 1, 0)
    # to get the correct (H, W, 2) layout. (hdf5storage auto-reverses dims,
    # but raw h5py does not.)
    with h5py.File(uvmap_path, "r") as f:
        uv = np.array(f["uv"], dtype=np.float64)  # (2, W, H) via h5py

    # uv[0] = u coordinates, uv[1] = v coordinates
    # Transpose from (2, W, H) to (H, W, 2)
    uv_hwc = np.transpose(uv, (2, 1, 0))  # (H, W, 2)

    # Create valid pixel mask BEFORE any NaN replacement
    valid_mask = (~np.isnan(uv_hwc).any(axis=-1)).astype(np.float32)  # (H, W), 1=valid, 0=NaN

    # Replace NaN with 0 temporarily for safe resize
    uv_hwc = np.nan_to_num(uv_hwc, nan=0.0)

    # Resize uvmap to match image native resolution if needed
    if uv_hwc.shape[:2] != (native_h, native_w):
        map_x = cv2.resize(uv_hwc[:, :, 0].astype(np.float32), (native_w, native_h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(uv_hwc[:, :, 1].astype(np.float32), (native_w, native_h), interpolation=cv2.INTER_LINEAR)
        uv_hwc_resized = np.stack([map_x, map_y], axis=-1)  # (H, W, 2)
        # Resize valid mask with NEAREST to preserve sharp boundary
        valid_mask = cv2.resize(valid_mask, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
    else:
        uv_hwc_resized = uv_hwc.astype(np.float32)

    # UVDoc uvmap is in [0, 1], convert to [-1, 1] for ForCenNet grid_sample
    backward_map = uv_hwc_resized * 2 - 1

    # Zero out invalid pixels: set BM to out-of-bounds value
    invalid = valid_mask < 0.5
    backward_map[invalid] = -5.0  # maps to pixel < 0, correctly out-of-bounds

    # 4. Detect lines from the distorted image at native resolution
    labels = generate_foreground_labels(image, include_lines=True, line_interval=line_interval)
    line_points_raw = labels.get('line_points', [])

    # 5. Resize all outputs to target image_size before saving
    if native_size is not None:
        image = cv2.resize(image, native_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, native_size, interpolation=cv2.INTER_NEAREST)
        valid_mask = cv2.resize(valid_mask, native_size, interpolation=cv2.INTER_NEAREST)
        bm_x = cv2.resize(backward_map[:, :, 0], native_size, interpolation=cv2.INTER_LINEAR)
        bm_y = cv2.resize(backward_map[:, :, 1], native_size, interpolation=cv2.INTER_LINEAR)
        backward_map = np.stack([bm_x, bm_y], axis=-1)
        # Re-apply mask after resize: any pixel where valid_mask < 0.5 -> out-of-bounds
        invalid = valid_mask < 0.5
        backward_map[invalid] = -5.0

        # Scale line_points coordinates from native to target resolution
        if len(line_points_raw) > 0 and (native_h != image_size or native_w != image_size):
            scale_x = image_size / native_w
            scale_y = image_size / native_h
            line_points_scaled = []
            for pts in line_points_raw:
                if len(pts) > 0:
                    scaled_pts = pts.copy().astype(np.float32)
                    scaled_pts[:, 0] *= scale_x
                    scaled_pts[:, 1] *= scale_y
                    line_points_scaled.append(scaled_pts)
                else:
                    line_points_scaled.append(pts.copy())
            line_points_raw = line_points_scaled

    # Save all outputs
    cv2.imwrite(os.path.join(output_dir, "distorted.png"), image)
    cv2.imwrite(os.path.join(output_dir, "mask.png"), mask)
    np.save(os.path.join(output_dir, "backward_map.npy"), backward_map.astype(np.float32))
    _save_line_points(output_dir, line_points_raw, interval=line_interval)


def _build_inverse_uvmap(uvmap, H, W, step=4):
    """
    Invert a forward UV map (distorted→canonical) into an inverse map
    (canonical→distorted) using subsampled griddata for speed.

    Args:
        uvmap: (H, W, 2) float32, UV map in [0,1], NaN for background.
        H, W: Full resolution dimensions.
        step: Subsampling factor for speed (default 4).

    Returns:
        inv_map: (H, W, 2) float32, pixel coordinates in the distorted
                 image for each canonical grid position.
    """
    uv_sub = uvmap[::step, ::step]
    Hs, Ws = uv_sub.shape[:2]

    js, is_ = np.meshgrid(
        np.arange(Ws, dtype=np.float64) * step,
        np.arange(Hs, dtype=np.float64) * step,
    )

    canon_u = uv_sub[:, :, 0].ravel()
    canon_v = uv_sub[:, :, 1].ravel()
    dist_x = js.ravel()
    dist_y = is_.ravel()

    valid = ~(np.isnan(canon_u) | np.isnan(canon_v))
    if np.sum(valid) < 4:
        # Not enough valid points; return identity mapping in pixel coords
        jj, ii = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        return np.stack([jj, ii], axis=-1)

    pts = np.column_stack([canon_u[valid], canon_v[valid]])
    vals_x = dist_x[valid]
    vals_y = dist_y[valid]

    u_coarse = np.linspace(0, 1, Ws)
    v_coarse = np.linspace(0, 1, Hs)
    Uc, Vc = np.meshgrid(u_coarse, v_coarse)

    inv_x_coarse = griddata(pts, vals_x, (Uc, Vc), method="linear", fill_value=np.nan)
    inv_y_coarse = griddata(pts, vals_y, (Uc, Vc), method="linear", fill_value=np.nan)

    nan_mask = np.isnan(inv_x_coarse) | np.isnan(inv_y_coarse)
    if np.any(nan_mask):
        inv_x_nn = griddata(pts, vals_x, (Uc, Vc), method="nearest")
        inv_y_nn = griddata(pts, vals_y, (Uc, Vc), method="nearest")
        inv_x_coarse[nan_mask] = inv_x_nn[nan_mask]
        inv_y_coarse[nan_mask] = inv_y_nn[nan_mask]

    inv_x = cv2.resize(inv_x_coarse.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    inv_y = cv2.resize(inv_y_coarse.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    return np.stack([inv_x, inv_y], axis=-1).astype(np.float32)


def convert_final_sample(
    img_path: str,
    seg_path: str,
    uvmap_path: str,
    output_dir: str,
    image_size: int = 288,
    line_interval: int = 4,
):
    """
    Convert a single create_final.py output sample to ForCenNet offline format.

    Key difference from convert_uvdoc_sample:
    - The image is a composited document (with texture + background), not raw.
    - The UV map is a forward map (distorted→canonical), so we must invert it
      to obtain the backward map (canonical→distorted) for ForCenNet training.

    Args:
        img_path: Path to composited image (PNG).
        seg_path: Path to segmentation mask (.mat, HDF5).
        uvmap_path: Path to UV mapping field (.mat, HDF5) — forward map.
        output_dir: Directory to save converted sample.
        image_size: Target image size (square) for saving output files.
        line_interval: Sampling interval for line control points.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read composited image
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    native_h, native_w = image.shape[:2]

    # 2. Read segmentation mask
    seg = h5.loadmat(seg_path)["seg"]  # (H, W)
    mask = (seg > 0).astype(np.uint8) * 255

    # 3. Read UV mapping field (forward: distorted→canonical)
    uvmap = h5.loadmat(uvmap_path)["uv"].astype(np.float32)  # (H, W, 2)

    # 4. Invert UV map to get backward map (canonical→distorted)
    # Build inverse map at full native resolution
    inv_map = _build_inverse_uvmap(uvmap, native_h, native_w, step=4)

    # inv_map contains (x, y) pixel coordinates in the distorted image
    # at native resolution. For ForCenNet training, the image will be resized
    # to image_size x image_size, so we must scale the inv_map coordinates
    # to the target resolution BEFORE normalizing to [-1, 1].
    # This ensures denormalization with (dim-1) gives correct pixel coords
    # in the resized image.
    inv_map_target = np.zeros_like(inv_map, dtype=np.float32)
    inv_map_target[:, :, 0] = inv_map[:, :, 0] / (native_w - 1) * (image_size - 1)
    inv_map_target[:, :, 1] = inv_map[:, :, 1] / (native_h - 1) * (image_size - 1)

    # Normalize to [-1, 1] using target dimensions
    backward_map = np.zeros_like(inv_map_target, dtype=np.float32)
    backward_map[:, :, 0] = inv_map_target[:, :, 0] / (image_size - 1) * 2 - 1
    backward_map[:, :, 1] = inv_map_target[:, :, 1] / (image_size - 1) * 2 - 1

    # NOTE: We do NOT apply the seg mask to backward_map here.
    # The seg mask marks foreground in the *distorted* image, but after
    # uvmap inversion, standard-space pixels that were background in the
    # distorted image may need to sample from foreground content
    # (e.g., when text is compressed and needs to expand upward).
    # build_inverse_uvmap already extrapolates valid coordinates to all
    # pixels via nearest-neighbor, so the backward map is complete.

    # 5. Detect lines from the composited image at native resolution
    labels = generate_foreground_labels(image, include_lines=True, line_interval=line_interval)
    line_points_raw = labels.get('line_points', [])

    # 6. Resize all outputs to target image_size before saving
    if image_size is not None:
        target_size = (image_size, image_size)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        bm_x = cv2.resize(backward_map[:, :, 0], target_size, interpolation=cv2.INTER_LINEAR)
        bm_y = cv2.resize(backward_map[:, :, 1], target_size, interpolation=cv2.INTER_LINEAR)
        backward_map = np.stack([bm_x, bm_y], axis=-1)

        # Scale line_points coordinates
        if len(line_points_raw) > 0 and (native_h != image_size or native_w != image_size):
            scale_x = image_size / native_w
            scale_y = image_size / native_h
            line_points_scaled = []
            for pts in line_points_raw:
                if len(pts) > 0:
                    scaled_pts = pts.copy().astype(np.float32)
                    scaled_pts[:, 0] *= scale_x
                    scaled_pts[:, 1] *= scale_y
                    line_points_scaled.append(scaled_pts)
                else:
                    line_points_scaled.append(pts.copy())
            line_points_raw = line_points_scaled

    # 7. Save all outputs
    cv2.imwrite(os.path.join(output_dir, "distorted.png"), image)
    cv2.imwrite(os.path.join(output_dir, "mask.png"), mask)
    np.save(os.path.join(output_dir, "backward_map.npy"), backward_map.astype(np.float32))
    _save_line_points(output_dir, line_points_raw, interval=line_interval)


def generate_from_clean_images(
    source_dir: str,
    output_dir: str,
    num_per_image: int = 50,
    image_size: Optional[int] = None,
    grid_size: int = 4,
    perturbation: float = 0.1,
):
    """
    Generate offline training data from clean document images using TPS distortion.

    Args:
        source_dir: Directory containing clean document images.
        output_dir: Directory to save generated samples.
        num_per_image: Number of distorted samples per source image.
        image_size: Target image size (square). If None, use native resolution of each source image.
        grid_size: TPS control point grid size.
        perturbation: TPS perturbation strength.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect source images
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    source_images = []
    for ext in extensions:
        source_images.extend(glob.glob(os.path.join(source_dir, ext)))
        source_images.extend(
            glob.glob(os.path.join(source_dir, "**", ext), recursive=True)
        )
    source_images = sorted(set(source_images))

    if len(source_images) == 0:
        raise RuntimeError(f"No images found in {source_dir}")

    size_str = f"{image_size}" if image_size is not None else "native (no resize)"
    print(f"Found {len(source_images)} source images.")
    print(f"Generating {num_per_image} samples per image...")
    print(f"Image size: {size_str}")
    print(f"Total samples: {len(source_images) * num_per_image}")

    sample_idx = 0
    line_interval = 4  # Control point sampling interval in pixels

    for img_path in tqdm(source_images, desc="Source images"):
        # Load clean image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Cannot read {img_path}, skipping.")
            continue

        # Resize to target size (or keep native resolution)
        if image_size is not None:
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        h, w = image.shape[:2]

        # Extract foreground labels (mask + line points) once per source image
        labels = generate_foreground_labels(image, include_lines=True, line_interval=line_interval)
        mask = labels['mask']  # (H, W) float32 {0, 1}
        clean_line_points = labels.get('line_points', [])  # list of (N_i, 2) arrays

        for i in range(num_per_image):
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx:06d}")
            os.makedirs(sample_dir, exist_ok=True)

            # Generate random TPS distortion
            forward_map, backward_map = generate_random_distortion_field(
                h, w, grid_size, perturbation, seed=None
            )

            # Warp image and mask
            distorted_image = warp_image(image, forward_map, mode="bilinear")
            distorted_mask = warp_mask(mask, forward_map)

            # Warp line points to get distorted control points
            if len(clean_line_points) > 0:
                distorted_line_points = []
                for pts in clean_line_points:
                    if len(pts) > 0:
                        warped_pts = warp_points(pts, forward_map, h, w)
                        distorted_line_points.append(warped_pts)
                    else:
                        distorted_line_points.append(pts.copy())
            else:
                distorted_line_points = []

            # Clip and save
            distorted_image = np.clip(distorted_image, 0, 255).astype(np.uint8)
            distorted_mask_binary = (distorted_mask > 0.5).astype(np.uint8) * 255

            cv2.imwrite(os.path.join(sample_dir, "distorted.png"), distorted_image)
            cv2.imwrite(os.path.join(sample_dir, "mask.png"), distorted_mask_binary)
            np.save(
                os.path.join(sample_dir, "backward_map.npy"),
                backward_map.astype(np.float32),
            )

            # Save distorted line control points
            _save_line_points(sample_dir, distorted_line_points, interval=line_interval)

            sample_idx += 1

    print(f"\nDone! Generated {sample_idx} samples in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate offline training data for ForCenNet"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["uvdoc", "synthesize", "uvdoc_final"],
        help="Data generation mode: 'uvdoc' (convert UVDoc_raw) or 'synthesize' (from clean images)",
    )
    parser.add_argument(
        "--uvdoc_dir",
        type=str,
        default=None,
        help="Path to UVDoc_raw directory (for 'uvdoc' mode)",
    )
    parser.add_argument(
        "--final_dir",
        type=str,
        default=None,
        help="Path to create_final.py output directory (for 'uvdoc_final' mode)",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Directory containing clean document images (for 'synthesize' mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--image_size", type=int, default=288,
        help="Target image size (square) for saving. Processing uses native resolution internally."
    )
    parser.add_argument(
        "--num_per_image",
        type=int,
        default=5,
        help="Number of samples per source image (for 'synthesize' mode)",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=4,
        help="TPS control point grid size (for 'synthesize' mode)",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.1,
        help="TPS perturbation strength (for 'synthesize' mode)",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.0,
        help="If > 0, split data into train/val with this ratio for validation set",
    )

    args = parser.parse_args()

    if args.mode == "uvdoc":
        if args.uvdoc_dir is None:
            print("Error: --uvdoc_dir is required for 'uvdoc' mode.")
            return

        rgb_dir = os.path.join(args.uvdoc_dir, "samples", "rgb")
        seg_dir = os.path.join(args.uvdoc_dir, "samples", "seg")
        uvmap_dir = os.path.join(args.uvdoc_dir, "samples", "uvmap")

        # Collect all sample IDs from rgb directory
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        if len(rgb_files) == 0:
            print(f"Error: No PNG files found in {rgb_dir}")
            return

        size_str = f"{args.image_size}" if args.image_size is not None else "native (no resize)"
        print(f"Found {len(rgb_files)} UVDoc_raw samples.")
        print(f"Output directory: {args.output_dir}")
        print(f"Image size: {size_str}")

        # Determine output subdirectories
        if args.split_ratio > 0:
            val_count = int(len(rgb_files) * args.split_ratio)
            train_count = len(rgb_files) - val_count
            print(f"Split: {train_count} train, {val_count} val")
            train_dir = os.path.join(args.output_dir, "train")
            val_dir = os.path.join(args.output_dir, "val")
        else:
            train_dir = args.output_dir
            val_dir = None

        sample_idx = 0
        for i, rgb_path in enumerate(tqdm(rgb_files, desc="Converting UVDoc")):
            basename = os.path.splitext(os.path.basename(rgb_path))[0]
            seg_path = os.path.join(seg_dir, f"{basename}.mat")
            uvmap_path = os.path.join(uvmap_dir, f"{basename}.mat")

            if not os.path.exists(seg_path):
                print(f"Warning: Missing seg file for {basename}, skipping.")
                continue
            if not os.path.exists(uvmap_path):
                print(f"Warning: Missing uvmap file for {basename}, skipping.")
                continue

            # Determine output directory
            if args.split_ratio > 0 and i >= train_count:
                dest_dir = os.path.join(val_dir, f"sample_{sample_idx:06d}")
            else:
                dest_dir = os.path.join(train_dir, f"sample_{sample_idx:06d}")

            try:
                convert_uvdoc_sample(
                    rgb_path, seg_path, uvmap_path,
                    dest_dir, args.image_size
                )
                sample_idx += 1
            except Exception as e:
                print(f"Warning: Error processing {basename}: {e}")

        print(f"\nDone! Converted {sample_idx} samples to: {args.output_dir}")

    elif args.mode == "synthesize":
        if args.source_dir is None:
            print("Error: --source_dir is required for 'synthesize' mode.")
            return

        if args.split_ratio > 0:
            # Split source images first, then generate
            extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
            source_images = []
            for ext in extensions:
                source_images.extend(glob.glob(os.path.join(args.source_dir, ext)))
                source_images.extend(
                    glob.glob(os.path.join(args.source_dir, "**", ext), recursive=True)
                )
            source_images = sorted(set(source_images))

            if len(source_images) == 0:
                print(f"Error: No images found in {args.source_dir}")
                return

            val_count = max(1, int(len(source_images) * args.split_ratio))
            train_count = len(source_images) - val_count
            print(f"Source split: {train_count} train images, {val_count} val images")

            # Create temporary source dirs for train/val
            import shutil
            tmp_dir = os.path.join(args.output_dir, "_tmp_sources")

            train_src_dir = os.path.join(tmp_dir, "train")
            val_src_dir = os.path.join(tmp_dir, "val")
            os.makedirs(train_src_dir, exist_ok=True)
            os.makedirs(val_src_dir, exist_ok=True)

            for i, src in enumerate(source_images[:train_count]):
                dst = os.path.join(train_src_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

            for i, src in enumerate(source_images[train_count:]):
                dst = os.path.join(val_src_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

            train_out = os.path.join(args.output_dir, "train")
            val_out = os.path.join(args.output_dir, "val")

            print(f"\n=== Generating training data ===")
            generate_from_clean_images(
                train_src_dir, train_out,
                args.num_per_image, args.image_size,
                args.grid_size, args.perturbation,
            )

            print(f"\n=== Generating validation data ===")
            generate_from_clean_images(
                val_src_dir, val_out,
                max(1, args.num_per_image // 5), args.image_size,
                args.grid_size, args.perturbation,
            )

            # Clean up temp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\nCleaned up temporary files.")

        else:
            generate_from_clean_images(
                args.source_dir, args.output_dir,
                args.num_per_image, args.image_size,
                args.grid_size, args.perturbation,
            )

    elif args.mode == "uvdoc_final":
        if args.final_dir is None:
            print("Error: --final_dir is required for 'uvdoc_final' mode.")
            return

        img_dir = os.path.join(args.final_dir, "img")
        seg_dir = os.path.join(args.final_dir, "seg")
        uvmap_dir = os.path.join(args.final_dir, "uvmap")
        meta_dir = os.path.join(args.final_dir, "metadata_sample")

        # Collect all image files
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if len(img_files) == 0:
            print(f"Error: No PNG images found in {img_dir}")
            return

        size_str = f"{args.image_size}" if args.image_size is not None else "native (no resize)"
        print(f"Found {len(img_files)} create_final.py samples.")
        print(f"Output directory: {args.output_dir}")
        print(f"Image size: {size_str}")

        sample_idx = 0
        for img_path in tqdm(img_files, desc="Converting final samples"):
            basename = os.path.splitext(os.path.basename(img_path))[0]  # e.g. "00000"

            # Load metadata to get geom_name (which is used for seg/uvmap file names)
            meta_path = os.path.join(meta_dir, f"{basename}.json")
            if not os.path.exists(meta_path):
                print(f"Warning: Missing metadata for {basename}, skipping.")
                continue

            with open(meta_path, "r") as f:
                meta = json.load(f)

            geom_name = meta["geom_name"]  # e.g. "00_00000_1_0_0"

            seg_path = os.path.join(seg_dir, f"{geom_name}.mat")
            uvmap_path = os.path.join(uvmap_dir, f"{geom_name}.mat")

            if not os.path.exists(seg_path):
                print(f"Warning: Missing seg file for {geom_name}, skipping.")
                continue
            if not os.path.exists(uvmap_path):
                print(f"Warning: Missing uvmap file for {geom_name}, skipping.")
                continue

            dest_dir = os.path.join(args.output_dir, f"sample_{sample_idx:06d}")

            try:
                convert_final_sample(
                    img_path, seg_path, uvmap_path,
                    dest_dir, args.image_size
                )
                sample_idx += 1
            except Exception as e:
                print(f"Warning: Error processing {basename}: {e}")

        print(f"\nDone! Converted {sample_idx} samples to: {args.output_dir}")


if __name__ == "__main__":
    main()

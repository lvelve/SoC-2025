"""
Dataset and DataLoader for ForCenNet.

Supports two modes:
1. Pre-generated data: Load pre-synthesized distorted/undistorted pairs from disk.
2. Online synthesis: Generate distortions on-the-fly from clean document images.
"""

import os
import json
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict

from .augmentation import (
    generate_training_pair,
    generate_random_distortion_field,
    warp_image,
    warp_mask,
    warp_points,
    RandomAugmentation,
)
from .label_generation import generate_foreground_labels


class ForCenNetDataset(Dataset):
    """
    Dataset for ForCenNet training.
    
    Operates in two modes:
    - 'online': Generates distorted pairs on-the-fly from clean document images.
    - 'offline': Loads pre-generated distorted pairs from disk.
    
    For 'online' mode, the data_dir should contain clean document images.
    For 'offline' mode, the data_dir should contain subdirectories with:
        - distorted.png / distorted.jpg
        - mask.png
        - backward_map.npy
    """
    
    def __init__(
        self,
        data_dir: str,
        mode: str = 'online',
        image_size: int = 288,
        num_synthetic: int = 50,
        tps_grid_size: int = 4,
        tps_perturbation: float = 0.1,
        use_augmentation: bool = True,
        include_lines: bool = True,
        line_interval: int = 4,
        transform=None
    ):
        """
        Args:
            data_dir: Path to data directory.
            mode: 'online' or 'offline'.
            image_size: Target image size (square).
            num_synthetic: Number of synthetic samples per source image (online mode).
            tps_grid_size: TPS control point grid size.
            tps_perturbation: TPS perturbation strength.
            use_augmentation: Whether to apply random augmentation.
            include_lines: Whether to extract line elements.
            line_interval: Line control point sampling interval.
            transform: Optional additional transform.
        """
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.num_synthetic = num_synthetic
        self.tps_grid_size = tps_grid_size
        self.tps_perturbation = tps_perturbation
        self.use_augmentation = use_augmentation
        self.include_lines = include_lines
        self.line_interval = line_interval
        self.transform = transform
        
        if use_augmentation:
            self.augmentor = RandomAugmentation()
        else:
            self.augmentor = None
        
        # Collect image paths
        self.image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        
        if mode == 'online':
            self._init_online_mode()
        elif mode == 'offline':
            self._init_offline_mode()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        print(f"[ForCenNetDataset] mode={mode}, samples={len(self)}")
    
    def _init_online_mode(self):
        """Initialize for online synthesis mode."""
        self.source_images = []
        for ext in self.image_extensions:
            self.source_images.extend(glob.glob(os.path.join(self.data_dir, ext)))
            self.source_images.extend(glob.glob(os.path.join(self.data_dir, '**', ext), recursive=True))
        
        # Remove duplicates and sort
        self.source_images = sorted(set(self.source_images))
        
        if len(self.source_images) == 0:
            raise RuntimeError(f"No images found in {self.data_dir}")
        
        # Total length = num_source_images * num_synthetic_per_image
        self.total_length = len(self.source_images) * self.num_synthetic
        
        # Cache for precomputed labels
        self._label_cache = {}
    
    def _init_offline_mode(self):
        """Initialize for offline pre-generated data mode."""
        # Expect subdirectories with samples
        self.sample_dirs = sorted([
            d for d in glob.glob(os.path.join(self.data_dir, '*'))
            if os.path.isdir(d)
        ])
        
        # If no subdirectories, try loading directly from data_dir
        if len(self.sample_dirs) == 0:
            # Check if data_dir itself contains the required files
            if os.path.exists(os.path.join(self.data_dir, 'backward_map.npy')):
                self.sample_dirs = [self.data_dir]
            else:
                raise RuntimeError(f"No valid samples found in {self.data_dir}")
        
        self.total_length = len(self.sample_dirs)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == 'online':
            return self._get_online_item(idx)
        else:
            return self._get_offline_item(idx)
    
    def _get_online_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a training sample on-the-fly."""
        # Determine which source image to use
        source_idx = idx % len(self.source_images)
        source_path = self.source_images[source_idx]
        
        # Load source image
        image = cv2.imread(source_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {source_path}")
        
        # Generate or retrieve foreground labels
        if source_idx not in self._label_cache:
            labels = generate_foreground_labels(
                image,
                include_lines=self.include_lines,
                line_interval=self.line_interval
            )
            self._label_cache[source_idx] = labels
        else:
            labels = self._label_cache[source_idx]
        
        mask = labels['mask']
        line_points = labels.get('sampled_points', None)
        
        # Generate training pair with random seed per sample
        seed = idx * 7 + 42  # Deterministic but varied
        pair = generate_training_pair(
            image=image,
            mask=mask,
            line_points=line_points,
            grid_size=self.tps_grid_size,
            perturbation=self.tps_perturbation,
            image_size=self.image_size,
            seed=None  # Use random for diversity
        )
        
        dist_image = pair['distorted_image']
        dist_mask = pair['distorted_mask']
        backward_map = pair['backward_map']
        
        # Apply augmentation
        if self.augmentor is not None:
            dist_image, dist_mask, backward_map = self.augmentor(
                dist_image, dist_mask, backward_map
            )
        
        # Convert to tensors
        # Image: (H, W, 3) uint8 -> (3, H, W) float32 [0, 1]
        img_tensor = torch.from_numpy(dist_image).float().permute(2, 0, 1) / 255.0
        
        # Mask: (H, W) -> (H, W) float32
        mask_tensor = torch.from_numpy(dist_mask).float()
        
        # Backward map: (H, W, 2) -> (2, H, W) float32
        bm_tensor = torch.from_numpy(backward_map).float().permute(2, 0, 1)
        
        # Get distorted line control points for curvature loss
        dist_line_points = pair.get('distorted_line_points', None)
        if dist_line_points is not None:
            # Pad line points to fixed size for batch collation
            line_points_tensor, line_lengths = self._pad_line_points(dist_line_points)
        else:
            line_points_tensor = torch.zeros(1, 2)
            line_lengths = torch.zeros(1, dtype=torch.long)
        
        result = {
            'image': img_tensor,           # (3, H, W)
            'mask_gt': mask_tensor,         # (H, W)
            'backward_map': bm_tensor,      # (2, H, W)
            'line_points': line_points_tensor,  # (max_lines, 2)
            'line_lengths': line_lengths,   # (max_lines,)
        }
        
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def _get_offline_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a pre-generated training sample from disk."""
        sample_dir = self.sample_dirs[idx]
        
        # Load distorted image
        img_path = self._find_file(sample_dir, ['distorted.png', 'distorted.jpg', 'image.png', 'image.jpg'])
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Record native resolution before resizing (for line_points coordinate scaling)
        native_h, native_w = image.shape[:2]
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Load mask
        mask_path = self._find_file(sample_dir, ['mask.png', 'mask.jpg'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Load backward mapping
        bm_path = os.path.join(sample_dir, 'backward_map.npy')
        if os.path.exists(bm_path):
            backward_map = np.load(bm_path)
            if backward_map.shape[0] != self.image_size:
                # Build valid mask from values (invalid pixels have BM <= -4)
                valid_mask = (backward_map[:, :, 0] > -4.0).astype(np.float32)
                # Resize mask with NEAREST to preserve sharp boundary
                valid_mask = cv2.resize(valid_mask, (self.image_size, self.image_size),
                                        interpolation=cv2.INTER_NEAREST)
                # Resize mapping field
                bm_h = cv2.resize(backward_map[:, :, 0], (self.image_size, self.image_size))
                bm_w = cv2.resize(backward_map[:, :, 1], (self.image_size, self.image_size))
                backward_map = np.stack([bm_h, bm_w], axis=-1)
                # Re-apply mask: pixels that were invalid stay out-of-bounds
                backward_map[valid_mask < 0.5] = -5.0
        else:
            # Generate identity mapping
            backward_map = self._identity_mapping()
        
        # Load line control points (if available)
        lp_path = os.path.join(sample_dir, 'line_points.npz')
        if os.path.exists(lp_path):
            lp_data = np.load(lp_path)
            all_points = lp_data['points']    # (N_total, 2)
            lengths = lp_data['lengths']       # (num_lines,)
            
            if len(lengths) > 0 and len(all_points) > 0:
                # Scale line_points if native resolution differs from target
                if native_h != self.image_size or native_w != self.image_size:
                    scale_x = self.image_size / native_w
                    scale_y = self.image_size / native_h
                    all_points = all_points.copy()
                    all_points[:, 0] *= scale_x
                    all_points[:, 1] *= scale_y
                
                # Reconstruct per-line arrays from concatenated points and lengths
                line_points_list = []
                offset = 0
                for length in lengths:
                    if length > 0:
                        line_points_list.append(all_points[offset:offset + length])
                        offset += length
                
                if len(line_points_list) > 0:
                    line_points_tensor, line_lengths = self._pad_line_points(line_points_list)
                else:
                    line_points_tensor = torch.zeros(1, 2)
                    line_lengths = torch.zeros(1, dtype=torch.long)
            else:
                line_points_tensor = torch.zeros(1, 2)
                line_lengths = torch.zeros(1, dtype=torch.long)
        else:
            line_points_tensor = torch.zeros(1, 2)
            line_lengths = torch.zeros(1, dtype=torch.long)
        
        # Apply augmentation
        if self.augmentor is not None:
            image, mask, backward_map = self.augmentor(image, mask, backward_map)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask).float()
        bm_tensor = torch.from_numpy(backward_map).float().permute(2, 0, 1)
        
        result = {
            'image': img_tensor,
            'mask_gt': mask_tensor,
            'backward_map': bm_tensor,
            'line_points': line_points_tensor,
            'line_lengths': line_lengths,
        }
        
        return result
    
    def _find_file(self, directory: str, candidates: List[str]) -> str:
        """Find the first existing file from candidates."""
        for name in candidates:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
        # Fallback: search for any image
        for ext in self.image_extensions:
            files = glob.glob(os.path.join(directory, ext))
            if files:
                return files[0]
        raise FileNotFoundError(f"No image found in {directory}")
    
    def _identity_mapping(self) -> np.ndarray:
        """Generate an identity (no-distortion) mapping field."""
        gx = np.linspace(-1, 1, self.image_size)
        gy = np.linspace(-1, 1, self.image_size)
        grid_x, grid_y = np.meshgrid(gx, gy)
        return np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    
    def _pad_line_points(
        self,
        line_points_list: List[np.ndarray],
        max_lines: int = 100,
        max_points: int = 500
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad line points to fixed size for batch collation.
        
        Args:
            line_points_list: List of (N_i, 2) arrays.
            max_lines: Maximum number of lines to keep.
            max_points: Maximum points per line.
        
        Returns:
            padded: (max_lines, 2) tensor with concatenated points.
            lengths: (max_lines,) tensor with point counts per line.
        """
        lengths = []
        all_points = []
        
        for i, pts in enumerate(line_points_list[:max_lines]):
            if len(pts) > max_points:
                # Subsample
                indices = np.linspace(0, len(pts) - 1, max_points, dtype=int)
                pts = pts[indices]
            lengths.append(len(pts))
            all_points.append(pts)
        
        # Pad to max_lines
        while len(lengths) < max_lines:
            lengths.append(0)
            all_points.append(np.zeros((1, 2), dtype=np.float32))
        
        # Concatenate all points
        if sum(lengths) > 0:
            all_pts = np.concatenate([p for p, l in zip(all_points, lengths) if l > 0], axis=0)
        else:
            all_pts = np.zeros((1, 2), dtype=np.float32)
        
        # Pad to fixed total size
        total_points = sum(lengths)
        if total_points < max_lines * 2:
            pad_size = max_lines * 2 - total_points
            all_pts = np.concatenate([all_pts, np.zeros((pad_size, 2), dtype=np.float32)], axis=0)
        
        padded = torch.from_numpy(all_pts[:max_lines * 2]).float()
        length_tensor = torch.tensor(lengths, dtype=torch.long)
        
        return padded, length_tensor


def create_dataloader(
    data_dir: str,
    mode: str = 'online',
    batch_size: int = 8,
    image_size: int = 288,
    num_synthetic: int = 50,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for ForCenNet training.
    
    Args:
        data_dir: Path to data directory.
        mode: 'online' or 'offline'.
        batch_size: Batch size.
        image_size: Target image size.
        num_synthetic: Number of synthetic samples per source image.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle.
        **kwargs: Additional arguments for ForCenNetDataset.
    
    Returns:
        DataLoader instance.
    """
    dataset = ForCenNetDataset(
        data_dir=data_dir,
        mode=mode,
        image_size=image_size,
        num_synthetic=num_synthetic,
        **kwargs
    )
    
    def collate_fn(batch):
        """Custom collate function to handle variable-size line points."""
        result = {
            'image': torch.stack([b['image'] for b in batch]),
            'mask_gt': torch.stack([b['mask_gt'] for b in batch]),
            'backward_map': torch.stack([b['backward_map'] for b in batch]),
        }
        # Line points are optional and may have variable sizes
        if 'line_points' in batch[0]:
            result['line_points'] = [b['line_points'] for b in batch]
            result['line_lengths'] = [b['line_lengths'] for b in batch]
        return result
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return loader
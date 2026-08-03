"""
Foreground label extraction for ForCenNet.

Provides:
- Foreground-background segmentation using thresholding or learned models
- Line element extraction using LSD and OCR
- Line control point sampling
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def extract_foreground_mask(
    image: np.ndarray,
    method: str = 'adaptive_threshold',
    block_size: int = 31,
    c_value: int = 10,
    morph_kernel_size: int = 3
) -> np.ndarray:
    """
    Extract foreground mask from a clean document image.
    
    For clean scanned documents, uses adaptive thresholding.
    For more complex images, a U-Net or Hi-SAM model would be preferred,
    but this provides a lightweight fallback.
    
    Args:
        image: (H, W, 3) BGR image, uint8.
        method: 'adaptive_threshold' or 'otsu'.
        block_size: Block size for adaptive thresholding.
        c_value: Constant subtracted from mean.
        morph_kernel_size: Kernel size for morphological operations.
    
    Returns:
        mask: (H, W) binary mask, 1 = foreground, 0 = background.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == 'adaptive_threshold':
        # Adaptive threshold works well for documents with varying illumination
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_value
        )
    elif method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    # Close small gaps in text
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    mask = (binary > 127).astype(np.float32)
    return mask


def extract_line_segments(
    image: np.ndarray,
    min_line_length: int = 30,
    max_line_gap: int = 10,
    hough_threshold: int = 50,
    filter_hv: bool = True,
    slope_threshold: float = 0.1
) -> List[np.ndarray]:
    """
    Extract line segments from a document image using Hough Line Transform.
    
    Falls back to HoughLinesP when LSD is not available.
    Filters for horizontal and vertical lines if requested.
    
    Args:
        image: (H, W, 3) BGR image.
        min_line_length: Minimum line length in pixels.
        max_line_gap: Maximum gap between line segments.
        hough_threshold: Hough transform accumulator threshold.
        filter_hv: If True, keep only horizontal/vertical lines.
        slope_threshold: Slope threshold for H/V filtering.
    
    Returns:
        line_points: List of (N, 2) arrays, each representing a line as a sequence of points.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []
    
    line_points_list = []
    for line in lines:
        x1, y1, x2, y2 = line.ravel()
        
        if filter_hv:
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-6:
                slope = float('inf')
            else:
                slope = abs(dy / dx)
            
            # Horizontal: slope < threshold, Vertical: slope > 1/threshold
            is_horizontal = slope < slope_threshold
            is_vertical = slope > (1.0 / slope_threshold) if slope != float('inf') else True
            
            if not (is_horizontal or is_vertical):
                continue
        
        # Generate points along the line (sample every pixel)
        num_points = max(int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2)), 2)
        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)
        points = np.stack([xs, ys], axis=-1).astype(np.float32)
        line_points_list.append(points)
    
    return line_points_list


def extract_text_line_boxes(
    image: np.ndarray,
    min_box_area: int = 100
) -> List[np.ndarray]:
    """
    Extract text line bounding boxes using connected component analysis.
    Uses the midline of each bounding box as the text line representation.
    
    This is a lightweight alternative to OCR-based text line extraction.
    
    Args:
        image: (H, W, 3) BGR image.
        min_box_area: Minimum area to keep a connected component.
    
    Returns:
        line_points: List of (N, 2) arrays, each being the midline of a text line.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilate to merge characters into text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    line_points_list = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_box_area:
            continue
        
        # Midline: horizontal line at the vertical center of the bounding box
        mid_y = y + h / 2.0
        num_points = max(w, 2)
        xs = np.linspace(x, x + w, num_points)
        ys = np.full_like(xs, mid_y)
        points = np.stack([xs, ys], axis=-1).astype(np.float32)
        line_points_list.append(points)
    
    return line_points_list


def sample_line_control_points(
    line_points: List[np.ndarray],
    interval: int = 4
) -> List[np.ndarray]:
    """
    Sample control points from line point sequences at regular intervals.
    
    Args:
        line_points: List of (N_i, 2) arrays of line points.
        interval: Sampling interval in pixels.
    
    Returns:
        sampled: List of (M_i, 2) arrays of sampled control points.
    """
    sampled_lines = []
    for pts in line_points:
        if len(pts) < 2:
            sampled_lines.append(pts)
            continue
        
        # Compute cumulative distance along the line
        diffs = np.diff(pts, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative[-1]
        
        if total_length < interval:
            sampled_lines.append(pts[:1])
            continue
        
        # Sample at regular intervals
        sample_dists = np.arange(0, total_length, interval)
        sampled_x = np.interp(sample_dists, cumulative, pts[:, 0])
        sampled_y = np.interp(sample_dists, cumulative, pts[:, 1])
        sampled = np.stack([sampled_x, sampled_y], axis=-1).astype(np.float32)
        sampled_lines.append(sampled)
    
    return sampled_lines


def generate_foreground_labels(
    image: np.ndarray,
    method: str = 'adaptive_threshold',
    include_lines: bool = True,
    line_interval: int = 4
) -> dict:
    """
    Generate all foreground labels from a clean document image.
    
    Args:
        image: (H, W, 3) BGR image, uint8.
        method: Segmentation method.
        include_lines: Whether to extract line elements.
        line_interval: Line control point sampling interval.
    
    Returns:
        dict with keys:
            'mask': (H, W) foreground mask
            'line_points': list of (N, 2) line point arrays
            'sampled_points': list of (M, 2) sampled control point arrays
    """
    # Extract foreground mask
    mask = extract_foreground_mask(image, method=method)
    
    result = {'mask': mask}
    
    if include_lines:
        # Extract line segments
        hough_lines = extract_line_segments(image)
        
        # Extract text line midlines
        text_lines = extract_text_line_boxes(image)
        
        # Combine all lines
        all_lines = hough_lines + text_lines
        
        # Sample control points
        sampled = sample_line_control_points(all_lines, interval=line_interval)
        
        result['line_points'] = all_lines
        result['sampled_points'] = sampled
    else:
        result['line_points'] = []
        result['sampled_points'] = []
    
    return result
"""
Inference script for ForCenNet.

Usage:
    python inference.py --image path/to/distorted.jpg --checkpoint checkpoints/best.pth --output result.png
    python inference.py --image path/to/distorted.jpg --checkpoint checkpoints/best.pth --output_dir ./results
    python inference.py --input_dir path/to/images/ --checkpoint checkpoints/best.pth --output_dir ./results
"""

import os
import sys
import argparse
import glob

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.forcenet import ForCenNet, build_forcenet
from utils.grid_sample import apply_backward_mapping
from utils.visualization import visualize_mapping_field, visualize_mask


def load_model(
    checkpoint_path: str,
    config: dict = None,
    device: torch.device = torch.device('cpu')
) -> ForCenNet:
    """
    Load ForCenNet model from checkpoint.
    
    Args:
        checkpoint_path: path to model checkpoint (.pth)
        config: model configuration dict (if None, uses defaults)
        device: torch device
    
    Returns:
        model: loaded ForCenNet model in eval mode
    """
    # Build model
    model = build_forcenet(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(
    image_path: str,
    target_size: int = 288,
    keep_aspect_ratio: bool = True
) -> tuple:
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path: path to input image
        target_size: target size (square)
        keep_aspect_ratio: whether to keep aspect ratio with padding
    
    Returns:
        image_tensor: (1, 3, target_size, target_size) tensor in [0, 1] for model input
        original_tensor: (1, 3, orig_h, orig_w) tensor in [0, 1] for high-res rectification
        original_size: (orig_h, orig_w) original image size
        pad_info: (top, bottom, left, right) padding applied
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    
    if keep_aspect_ratio:
        # Resize keeping aspect ratio, then pad to square
        scale = min(target_size / orig_h, target_size / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target_size x target_size
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        image_padded = cv2.copyMakeBorder(
            image_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        pad_info = (top, bottom, left, right)
    else:
        image_padded = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        pad_info = (0, 0, 0, 0)
    
    # Model input tensor (padded/resized to target_size)
    image_tensor = torch.from_numpy(image_padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Original image tensor for high-res rectification
    original_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    return image_tensor, original_tensor, (orig_h, orig_w), pad_info


def tensor_to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, 3, H, W) tensor in [0, 1] to (H, W, 3) BGR uint8 numpy array.
    
    Args:
        image_tensor: (1, 3, H, W) tensor in [0, 1]
    
    Returns:
        output: (H, W, 3) BGR uint8 numpy array
    """
    output = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def run_inference(
    model: ForCenNet,
    image_path: str,
    output_path: str = None,
    output_dir: str = None,
    target_size: int = 288,
    device: torch.device = torch.device('cpu'),
    save_mapping: bool = False,
    save_mask: bool = False,
    keep_aspect_ratio: bool = True
) -> dict:
    """
    Run inference on a single image.
    
    The model predicts a backward mapping (BM) at low resolution (target_size x target_size).
    For high-quality rectification, the BM is:
    1. Cropped to remove padding
    2. Bilinearly interpolated to the original image resolution
    3. Applied to the original high-resolution image via grid_sample
    
    This avoids quality loss from rectifying at low resolution then upscaling.
    
    Args:
        model: loaded ForCenNet model
        image_path: path to input distorted image
        output_path: path to save rectified image
        output_dir: directory to save outputs (alternative to output_path)
        target_size: inference resolution for the model
        device: torch device
        save_mapping: whether to save the mapping field visualization
        save_mask: whether to save the foreground mask visualization
        keep_aspect_ratio: whether to keep aspect ratio with padding
    
    Returns:
        dict with results
    """
    # Preprocess: get model input tensor and original high-res tensor
    image_tensor, original_tensor, original_size, pad_info = preprocess_image(
        image_path, target_size, keep_aspect_ratio
    )
    image_tensor = image_tensor.to(device)
    original_tensor = original_tensor.to(device)
    
    # Model inference at low resolution
    with torch.no_grad():
        output = model.rectify(image_tensor)
    
    bm_pred = output['bm_pred']          # (1, 2, target_size, target_size)
    mask_logits = output['mask_logits']   # (1, 2, target_size, target_size)
    
    # --- High-resolution rectification ---
    # 1. Crop BM to remove padding (keep only the content region)
    #    and remap coordinate values from padded space to content space
    top, bottom, left, right = pad_info
    bm_h = target_size - top - bottom
    bm_w = target_size - left - right
    
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        bm_cropped = bm_pred[:, :, top:top+bm_h, left:left+bm_w]
        # BM values reference positions in the padded [-1,1] space.
        # After cropping, we need to remap to the content-only [-1,1] space.
        # In padded pixel coords: px = (val + 1) / 2 * target_size
        # In content pixel coords: cpx = px - left (or top for y)
        # Back to [-1,1]: new_val = cpx / bm_w * 2 - 1
        # Combined: new_val = ((val + 1) / 2 * target_size - offset) / new_size * 2 - 1
        #         = (val + 1) * (target_size / new_size) - 2 * offset / new_size - 1
        scale_x = target_size / bm_w
        scale_y = target_size / bm_h
        bm_cropped = bm_cropped.clone()
        bm_cropped[:, 0, :, :] = (bm_cropped[:, 0, :, :] + 1.0) * scale_x - (2.0 * left / bm_w + 1.0)
        bm_cropped[:, 1, :, :] = (bm_cropped[:, 1, :, :] + 1.0) * scale_y - (2.0 * top / bm_h + 1.0)
    else:
        bm_cropped = bm_pred
    
    # 2. Resize BM to original image resolution (bilinear interpolation of the continuous field)
    orig_h, orig_w = original_size
    bm_resized = F.interpolate(bm_cropped, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
    
    # 3. Apply resized BM to original high-resolution image
    rectified_hires = apply_backward_mapping(original_tensor, bm_resized, mode='bilinear', padding_mode='border', align_corners=True)
    
    # 4. Convert to numpy
    result_image = tensor_to_numpy_image(rectified_hires)
    
    # Determine output path
    if output_path is None:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            basename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f'{basename}_rectified.png')
        else:
            base, ext = os.path.splitext(image_path)
            output_path = f'{base}_rectified{ext}'
    
    # Save rectified image
    cv2.imwrite(output_path, result_image)
    
    results = {
        'rectified_path': output_path,
        'rectified_image': result_image,
    }
    
    # Save mapping visualization (at model resolution)
    if save_mapping and output_dir:
        mapping_vis = visualize_mapping_field(bm_pred[0])
        mapping_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_mapping.png')
        cv2.imwrite(mapping_path, mapping_vis)
        results['mapping_path'] = mapping_path
    
    # Save mask visualization (at model resolution)
    if save_mask and output_dir:
        mask_prob = F.softmax(mask_logits[0], dim=0)
        mask_fg = mask_prob[1]
        mask_vis = visualize_mask(mask_fg)
        mask_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_mask.png')
        cv2.imwrite(mask_path, mask_vis)
        results['mask_path'] = mask_path
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ForCenNet Inference - Document Image Rectification')
    parser.add_argument('--image', type=str, default=None, help='Path to single input image')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory of input images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Path to save output image (single image mode)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--image_size', type=int, default=288, help='Inference image size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda or cpu)')
    parser.add_argument('--save_mapping', action='store_true', help='Save mapping field visualization')
    parser.add_argument('--save_mask', action='store_true', help='Save foreground mask visualization')
    parser.add_argument('--no_keep_aspect', action='store_true', help='Do not keep aspect ratio')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    print("Model loaded successfully.")
    
    # Default output directory
    if args.output_dir is None and args.output is None:
        args.output_dir = './inference_results'
    
    keep_aspect = not args.no_keep_aspect
    
    # Single image mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        print(f"Processing: {args.image}")
        results = run_inference(
            model=model,
            image_path=args.image,
            output_path=args.output,
            output_dir=args.output_dir,
            target_size=args.image_size,
            device=device,
            save_mapping=args.save_mapping,
            save_mask=args.save_mask,
            keep_aspect_ratio=keep_aspect
        )
        print(f"Rectified image saved to: {results['rectified_path']}")
    
    # Batch mode
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            return
        
        # Collect images
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input_dir, '**', ext), recursive=True))
        
        image_paths = sorted(set(image_paths))
        
        if len(image_paths) == 0:
            print(f"No images found in {args.input_dir}")
            return
        
        print(f"Found {len(image_paths)} images to process.")
        
        if args.output_dir is None:
            args.output_dir = os.path.join(args.input_dir, 'results')
        os.makedirs(args.output_dir, exist_ok=True)
        
        for i, img_path in enumerate(image_paths):
            try:
                print(f"[{i+1}/{len(image_paths)}] Processing: {img_path}")
                results = run_inference(
                    model=model,
                    image_path=img_path,
                    output_dir=args.output_dir,
                    target_size=args.image_size,
                    device=device,
                    save_mapping=args.save_mapping,
                    save_mask=args.save_mask,
                    keep_aspect_ratio=keep_aspect
                )
                print(f"  -> Saved: {results['rectified_path']}")
            except Exception as e:
                print(f"  -> Error: {e}")
        
        print(f"\nAll results saved to: {args.output_dir}")
    
    else:
        print("Error: Please provide --image or --input_dir")
        parser.print_help()


if __name__ == '__main__':
    main()
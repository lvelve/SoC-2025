"""
Training script for ForCenNet.

Usage:
    python train.py --config configs/default.yaml --data_dir ./data/train
    python train.py --data_dir ./data/train --epochs 30 --batch_size 8
    python train.py --data_dir ./data/train --mode offline  # for pre-generated data

For quick testing with synthetic data:
    python train.py --data_dir ./data/train --epochs 1 --batch_size 2 --num_workers 0
"""

import os
import sys
import time
import argparse
import yaml
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.forcenet import ForCenNet, build_forcenet
from data.dataset import ForCenNetDataset, create_dataloader
from losses.segmentation_loss import SegmentationLoss
from losses.mapping_loss import MappingLoss
from losses.curvature_loss import CurvatureLoss
from utils.grid_sample import apply_backward_mapping, compute_mapping_error
from utils.visualization import save_training_visualization

import numpy as np

# Memory optimization: enable TF32 for RTX GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def setup_logging(save_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger('ForCenNet')
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    log_file = os.path.join(save_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return config


def create_model(config: dict, device: torch.device) -> ForCenNet:
    """Create and initialize ForCenNet model."""
    model = build_forcenet(config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def create_optimizer(model: nn.Module, config: dict):
    """Create optimizer and scheduler."""
    train_config = config.get('training', {})
    
    lr = train_config.get('learning_rate', 1e-4)
    weight_decay = train_config.get('weight_decay', 0.01)
    beta1 = train_config.get('beta1', 0.9)
    beta2 = train_config.get('beta2', 0.999)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )
    
    return optimizer


def create_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Create learning rate scheduler."""
    train_config = config.get('training', {})
    
    epochs = train_config.get('epochs', 30)
    warmup_ratio = train_config.get('warmup_ratio', 0.1)
    max_lr = train_config.get('learning_rate', 1e-4)
    
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=warmup_ratio,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=10000
    )
    
    return scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    seg_criterion,
    map_criterion,
    curv_criterion,
    device: torch.device,
    epoch: int,
    config: dict,
    logger: logging.Logger,
    scaler: GradScaler = None,
    vis_dir: str = './vis',
    accum_steps: int = 1
) -> dict:
    """Train for one epoch."""
    model.train()
    
    train_config = config.get('training', {})
    loss_config = config.get('losses', {})
    log_interval = train_config.get('log_interval', 10)
    use_amp = train_config.get('use_amp', False)
    
    seg_weight = loss_config.get('seg_weight', 1.0)
    map_weight = loss_config.get('map_weight', 1.0)
    curv_weight = loss_config.get('curvature_weight', 1.0)
    
    data_config = config.get('data', {})
    image_size = data_config.get('image_size', 288)
    
    total_loss = 0.0
    total_seg_loss = 0.0
    total_map_loss = 0.0
    total_curv_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        images = batch['image'].to(device)           # (B, 3, H, W)
        mask_gt = batch['mask_gt'].to(device)         # (B, H, W)
        bm_gt = batch['backward_map'].to(device)      # (B, 2, H, W)
        
        # Get line points for curvature loss (optional)
        line_points = batch.get('line_points', None)
        
        # Zero gradients at the start of each accumulation cycle
        if (batch_idx % accum_steps) == 0:
            optimizer.zero_grad()
        
        # Forward pass
        if use_amp and scaler is not None:
            with autocast('cuda'):
                output = model(images, return_mask=True)
                bm_pred = output['bm_pred']
                mask_logits = output['mask_logits']
                
                # Compute losses
                loss_seg = seg_criterion(mask_logits, mask_gt)
                loss_map = map_criterion(bm_pred, bm_gt)
                loss_curv = curv_criterion(bm_pred, bm_gt, line_points, image_size)
                
                loss = seg_weight * loss_seg + map_weight * loss_map + curv_weight * loss_curv
                loss = loss / accum_steps  # Scale loss for accumulation
            
            scaler.scale(loss).backward()
            
            # Step optimizer only after accumulating enough gradients
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        else:
            output = model(images, return_mask=True)
            bm_pred = output['bm_pred']
            mask_logits = output['mask_logits']
            
            # Compute losses
            loss_seg = seg_criterion(mask_logits, mask_gt)
            loss_map = map_criterion(bm_pred, bm_gt)
            loss_curv = curv_criterion(bm_pred, bm_gt, line_points, image_size)
            
            loss = seg_weight * loss_seg + map_weight * loss_map + curv_weight * loss_curv
            loss = loss / accum_steps  # Scale loss for accumulation
            
            loss.backward()
            
            # Step optimizer only after accumulating enough gradients
            if (batch_idx + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        
        # Use unscaled loss for logging
        loss_item = loss.item() * accum_steps
        
        # Accumulate losses (use unscaled values)
        total_loss += loss_item
        total_seg_loss += loss_seg.item()
        total_map_loss += loss_map.item()
        total_curv_loss += loss_curv.item()
        num_batches += 1
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_seg = total_seg_loss / num_batches
            avg_map = total_map_loss / num_batches
            avg_curv = total_curv_loss / num_batches
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.6f} (Seg: {avg_seg:.6f}, Map: {avg_map:.6f}, Curv: {avg_curv:.6f}) "
                f"LR: {lr_now:.2e} Time: {elapsed:.1f}s"
            )
        
        # Save visualization periodically
        if (batch_idx + 1) % (log_interval * 10) == 0:
            try:
                with torch.no_grad():
                    rect_output = model.rectify(images[:1])
                    save_training_visualization(
                        epoch=epoch,
                        iteration=batch_idx,
                        distorted=images[0],
                        rectified=rect_output['rectified'][0],
                        bm_pred=bm_pred[0],
                        mask_logits=mask_logits[0],
                        bm_gt=bm_gt[0],
                        save_dir=vis_dir
                    )
            except Exception as e:
                logger.warning(f"Visualization save failed: {e}")
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_seg = total_seg_loss / max(num_batches, 1)
    avg_map = total_map_loss / max(num_batches, 1)
    avg_curv = total_curv_loss / max(num_batches, 1)
    
    return {
        'loss': avg_loss,
        'seg_loss': avg_seg,
        'map_loss': avg_map,
        'curv_loss': avg_curv
    }


def validate(
    model: nn.Module,
    dataloader,
    seg_criterion,
    map_criterion,
    curv_criterion,
    device: torch.device,
    config: dict
) -> dict:
    """Validate the model."""
    model.eval()
    
    loss_config = config.get('losses', {})
    seg_weight = loss_config.get('seg_weight', 1.0)
    map_weight = loss_config.get('map_weight', 1.0)
    curv_weight = loss_config.get('curvature_weight', 1.0)
    data_config = config.get('data', {})
    image_size = data_config.get('image_size', 288)
    
    total_loss = 0.0
    total_map_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            mask_gt = batch['mask_gt'].to(device)
            bm_gt = batch['backward_map'].to(device)
            line_points = batch.get('line_points', None)
            
            output = model(images, return_mask=True)
            bm_pred = output['bm_pred']
            mask_logits = output['mask_logits']
            
            loss_seg = seg_criterion(mask_logits, mask_gt)
            loss_map = map_criterion(bm_pred, bm_gt)
            loss_curv = curv_criterion(bm_pred, bm_gt, line_points, image_size)
            
            loss = seg_weight * loss_seg + map_weight * loss_map + curv_weight * loss_curv
            
            # Compute mapping error metrics
            metrics = compute_mapping_error(bm_pred, bm_gt)
            
            total_loss += loss.item()
            total_map_mae += metrics['mae']
            num_batches += 1
    
    return {
        'val_loss': total_loss / max(num_batches, 1),
        'val_map_mae': total_map_mae / max(num_batches, 1)
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    loss: float,
    save_path: str
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train ForCenNet')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default=None, help='Path to validation data directory')
    parser.add_argument('--mode', type=str, default='online', choices=['online', 'offline'],
                       help='Data loading mode: online (synthesize on-the-fly) or offline (pre-generated)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=None, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--vis_dir', type=str, default=None, help='Directory to save visualizations')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda or cpu)')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr is not None:
        config.setdefault('training', {})['learning_rate'] = args.lr
    if args.image_size is not None:
        config.setdefault('data', {})['image_size'] = args.image_size
    if args.num_workers is not None:
        config.setdefault('data', {})['num_workers'] = args.num_workers
    if args.save_dir is not None:
        config.setdefault('training', {})['save_dir'] = args.save_dir
    if args.use_amp:
        config.setdefault('training', {})['use_amp'] = True
    config.setdefault('training', {})['accum_steps'] = args.accum_steps
    
    train_config = config.get('training', {})
    data_config = config.get('data', {})
    
    # Setup
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    save_dir = train_config.get('save_dir', './checkpoints')
    vis_dir = args.vis_dir or os.path.join(save_dir, 'vis')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir)
    logger.info("=" * 60)
    logger.info("ForCenNet Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Mode: {args.mode}")
    
    # Create data loaders
    batch_size = train_config.get('batch_size', 8)
    image_size = data_config.get('image_size', 288)
    num_workers = data_config.get('num_workers', 4)
    num_synthetic = data_config.get('num_synthetic_per_image', 50)
    
    logger.info("Creating data loaders...")
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        mode=args.mode,
        batch_size=batch_size,
        image_size=image_size,
        num_synthetic=num_synthetic,
        num_workers=num_workers,
        shuffle=True
    )
    
    val_loader = None
    if args.val_dir and os.path.exists(args.val_dir):
        val_loader = create_dataloader(
            data_dir=args.val_dir,
            mode=args.mode,
            batch_size=batch_size,
            image_size=image_size,
            num_synthetic=1,  # Fewer for validation
            num_workers=num_workers,
            shuffle=False
        )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, device)
    
    # Gradient accumulation setup
    accum_steps = train_config.get('accum_steps', 1)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    # Scheduler total_steps should account for effective steps (optimizer steps per epoch)
    effective_steps_per_epoch = len(train_loader) // accum_steps
    scheduler = create_scheduler(optimizer, config, effective_steps_per_epoch)
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")
    
    # Create loss functions
    seg_criterion = SegmentationLoss()
    map_criterion = MappingLoss()
    curv_criterion = CurvatureLoss(
        epsilon=config.get('losses', {}).get('curvature_epsilon', 1e-4),
        sample_interval=config.get('losses', {}).get('line_sample_interval', 4)
    )
    
    # Mixed precision scaler
    use_amp = train_config.get('use_amp', False)
    scaler = GradScaler('cuda') if use_amp else None
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # Training loop
    epochs = train_config.get('epochs', 30)
    save_interval = train_config.get('save_interval', 5)
    
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Image size: {image_size}")
    logger.info(f"Accumulation steps: {accum_steps}, Effective batch: {batch_size * accum_steps}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"\n{'='*40} Epoch {epoch + 1}/{epochs} {'='*40}")
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            seg_criterion=seg_criterion,
            map_criterion=map_criterion,
            curv_criterion=curv_criterion,
            device=device,
            epoch=epoch,
            config=config,
            logger=logger,
            scaler=scaler,
            vis_dir=vis_dir,
            accum_steps=accum_steps
        )
        
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_metrics['loss']:.6f} "
            f"(Seg: {train_metrics['seg_loss']:.6f}, "
            f"Map: {train_metrics['map_loss']:.6f}, "
            f"Curv: {train_metrics['curv_loss']:.6f})"
        )
        
        # Validate
        if val_loader is not None:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                seg_criterion=seg_criterion,
                map_criterion=map_criterion,
                curv_criterion=curv_criterion,
                device=device,
                config=config
            )
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Val Loss: {val_metrics['val_loss']:.6f} "
                f"Val Map MAE: {val_metrics['val_map_mae']:.6f}"
            )
            current_loss = val_metrics['val_loss']
        else:
            current_loss = train_metrics['loss']
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch{epoch + 1:03d}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, current_loss, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            best_path = os.path.join(save_dir, 'best.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_loss, best_path)
            logger.info(f"Saved best model with loss: {best_loss:.6f}")
        
        # Save latest
        latest_path = os.path.join(save_dir, 'latest.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, current_loss, latest_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info(f"Checkpoints saved to: {save_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
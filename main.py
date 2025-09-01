#!/usr/bin/env python3
"""
Main training script for worm segmentation
Usage: python main.py [--config-options]
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import wandb

from config import Config
from trainer import train_model
from dataset import create_data_loaders, visualize_batch
from inference import run_inference_on_folder


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Worm Segmentation Training')
    
    # Data paths
    parser.add_argument('--data-root', type=str, 
                       default="../supervisely_download",
                       help='Root directory containing images and masks')
    parser.add_argument('--img-dir', type=str, default='images',
                       help='Image directory name')
    parser.add_argument('--mask-dir', type=str, default='masks_png',
                       help='Mask directory name')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='UnetPlusPlus',
                       choices=['UnetPlusPlus', 'DeepLabV3Plus', 'Unet', 'FPN', 'PSPNet', 'MAnet'],
                       help='Model architecture')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4',
                       help='Backbone encoder')
    parser.add_argument('--img-size', type=int, nargs=2, default=[512, 512],
                       help='Image size for training (height width)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Loss and optimization
    parser.add_argument('--loss', type=str, default='dice_focal',
                       choices=['dice', 'focal', 'dice_focal', 'tversky', 'iou', 'bce', 'combined'],
                       help='Loss function')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='Learning rate scheduler')
    
    # Augmentation and training settings
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--no-tta', action='store_true',
                       help='Disable test-time augmentation')
    
    # Execution modes
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'visualize', 'inference'],
                       help='Execution mode')
    parser.add_argument('--checkpoint', type=str,
                       help='Checkpoint path for inference mode')
    parser.add_argument('--input-folder', type=str,
                       help='Input folder for inference mode')
    parser.add_argument('--output-folder', type=str, default='inference_results',
                       help='Output folder for inference mode')
    
    # Other options
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    return parser.parse_args()


def update_config_from_args(config: Config, args) -> Config:
    """Update configuration with command line arguments"""
    
    # Data paths
    config.data_root = args.data_root
    config.img_dir = args.img_dir
    config.mask_dir = args.mask_dir
    
    # Model configuration
    config.model_name = args.model
    config.encoder_name = args.encoder
    config.img_size = tuple(args.img_size)
    
    # Training parameters
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    
    # Loss and optimization
    config.loss_function = args.loss
    config.optimizer = args.optimizer
    config.scheduler = args.scheduler
    
    # Settings
    config.use_augmentation = not args.no_augmentation
    config.use_wandb = not args.no_wandb
    config.test_time_augmentation = not args.no_tta
    config.save_dir = args.save_dir
    config.num_workers = args.num_workers
    
    return config


def check_data_availability(config: Config):
    """Check if data is available and accessible"""
    
    print("Checking data availability...")
    
    # Check if directories exist
    if not os.path.exists(config.img_path):
        print(f"❌ Image directory not found: {config.img_path}")
        return False
    
    if not os.path.exists(config.mask_path):
        print(f"❌ Mask directory not found: {config.mask_path}")
        return False
    
    # Count images and masks
    image_files = [f for f in os.listdir(config.img_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    mask_files = [f for f in os.listdir(config.mask_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"✅ Found {len(image_files)} images in {config.img_path}")
    print(f"✅ Found {len(mask_files)} masks in {config.mask_path}")
    
    if len(image_files) == 0:
        print("❌ No images found!")
        return False
    
    if len(mask_files) == 0:
        print("❌ No masks found!")
        return False
    
    return True


def visualize_data(config: Config):
    """Visualize sample data"""
    print("Creating data loaders for visualization...")
    
    try:
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        print("\n=== Training Data Sample ===")
        visualize_batch(train_loader, config, num_samples=4)
        
        print("\n=== Validation Data Sample ===")
        visualize_batch(val_loader, config, num_samples=4)
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {str(e)}")
        return False
    
    return True


def main():
    """Main function"""
    
    print("🐛 Worm Segmentation Training Pipeline")
    print("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Create and update configuration
    config = Config()
    config = update_config_from_args(config, args)
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Encoder: {config.encoder_name}")
    print(f"   Image size: {config.img_size}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Loss function: {config.loss_function}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Data augmentation: {config.use_augmentation}")
    print(f"   Wandb logging: {config.use_wandb}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Execute based on mode
    if args.mode == 'train':
        # Check data availability
        if not check_data_availability(config):
            print("❌ Data check failed. Please verify your data paths.")
            sys.exit(1)
        
        print(f"\n🚀 Starting training...")
        try:
            model, trainer = train_model(config)
            print("✅ Training completed successfully!")
            
            # Print best metrics
            if hasattr(trainer.callback_metrics, 'val_dice'):
                print(f"   Best validation Dice: {trainer.callback_metrics['val_dice']:.4f}")
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            sys.exit(1)
    
    elif args.mode == 'visualize':
        # Check data availability
        if not check_data_availability(config):
            print("❌ Data check failed. Please verify your data paths.")
            sys.exit(1)
        
        print(f"\n👁️  Visualizing data...")
        if not visualize_data(config):
            sys.exit(1)
    
    elif args.mode == 'inference':
        if not args.checkpoint:
            print("❌ Checkpoint path required for inference mode")
            sys.exit(1)
        
        if not args.input_folder:
            print("❌ Input folder required for inference mode")
            sys.exit(1)
        
        if not os.path.exists(args.checkpoint):
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        
        if not os.path.exists(args.input_folder):
            print(f"❌ Input folder not found: {args.input_folder}")
            sys.exit(1)
        
        print(f"\n🔮 Running inference...")
        print(f"   Model: {args.checkpoint}")
        print(f"   Input: {args.input_folder}")
        print(f"   Output: {args.output_folder}")
        
        try:
            run_inference_on_folder(
                model_path=args.checkpoint,
                config=config,
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                threshold=0.5,
                use_tta=config.test_time_augmentation,
                apply_postprocessing=True
            )
            print("✅ Inference completed successfully!")
            
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")
            sys.exit(1)
    
    print("\n🎉 All done!")


if __name__ == "__main__":
    main() 
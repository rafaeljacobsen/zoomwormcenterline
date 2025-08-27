#!/usr/bin/env python3
"""
Data exploration script for worm segmentation dataset
Usage: python explore_data.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from collections import Counter
from typing import List, Tuple, Dict
import argparse

from config import Config


def analyze_images(img_dir: str) -> Dict:
    """Analyze image characteristics"""
    
    print(f"Analyzing images in: {img_dir}")
    
    image_files = [f for f in os.listdir(img_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("No images found!")
        return {}
    
    print(f"Found {len(image_files)} images")
    
    # Analyze subset of images
    sample_size = min(100, len(image_files))
    sample_files = np.random.choice(image_files, sample_size, replace=False)
    
    widths, heights, channels, file_sizes = [], [], [], []
    
    for img_file in sample_files:
        img_path = os.path.join(img_dir, img_file)
        
        try:
            # Get file size
            file_size = os.path.getsize(img_path) / 1024  # KB
            file_sizes.append(file_size)
            
            # Load image
            img = cv2.imread(img_path)
            if img is not None:
                h, w, c = img.shape
                heights.append(h)
                widths.append(w)
                channels.append(c)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    stats = {
        'count': len(image_files),
        'sample_analyzed': len(widths),
        'width_stats': {
            'min': min(widths) if widths else 0,
            'max': max(widths) if widths else 0,
            'mean': np.mean(widths) if widths else 0,
            'std': np.std(widths) if widths else 0
        },
        'height_stats': {
            'min': min(heights) if heights else 0,
            'max': max(heights) if heights else 0,
            'mean': np.mean(heights) if heights else 0,
            'std': np.std(heights) if heights else 0
        },
        'channels': Counter(channels),
        'file_size_stats': {
            'min': min(file_sizes) if file_sizes else 0,
            'max': max(file_sizes) if file_sizes else 0,
            'mean': np.mean(file_sizes) if file_sizes else 0,
            'std': np.std(file_sizes) if file_sizes else 0
        },
        'aspect_ratios': [w/h for w, h in zip(widths, heights)] if widths and heights else []
    }
    
    return stats


def analyze_masks(mask_dir: str) -> Dict:
    """Analyze mask characteristics"""
    
    print(f"Analyzing masks in: {mask_dir}")
    
    mask_files = [f for f in os.listdir(mask_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not mask_files:
        print("No masks found!")
        return {}
    
    print(f"Found {len(mask_files)} masks")
    
    # Analyze subset of masks
    sample_size = min(50, len(mask_files))
    sample_files = np.random.choice(mask_files, sample_size, replace=False)
    
    mask_coverage = []  # Percentage of foreground pixels
    unique_values = []
    
    for mask_file in sample_files:
        mask_path = os.path.join(mask_dir, mask_file)
        
        try:
            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Calculate mask coverage
                total_pixels = mask.shape[0] * mask.shape[1]
                foreground_pixels = np.sum(mask > 0)
                coverage = (foreground_pixels / total_pixels) * 100
                mask_coverage.append(coverage)
                
                # Get unique values
                unique_vals = np.unique(mask)
                unique_values.extend(unique_vals.tolist())
                
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
    
    stats = {
        'count': len(mask_files),
        'sample_analyzed': len(mask_coverage),
        'coverage_stats': {
            'min': min(mask_coverage) if mask_coverage else 0,
            'max': max(mask_coverage) if mask_coverage else 0,
            'mean': np.mean(mask_coverage) if mask_coverage else 0,
            'std': np.std(mask_coverage) if mask_coverage else 0
        },
        'unique_values': sorted(list(set(unique_values))),
        'value_counts': Counter(unique_values)
    }
    
    return stats


def check_image_mask_pairs(img_dir: str, mask_dir: str) -> Dict:
    """Check correspondence between images and masks"""
    
    print("Checking image-mask pairs...")
    
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
    
    mask_files = [f for f in os.listdir(mask_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Find matching pairs
    pairs = []
    orphaned_images = []
    orphaned_masks = []
    
    for img_name, img_file in image_files.items():
        # Find corresponding masks (masks may have additional suffixes)
        matching_masks = [m for m in mask_files if m.startswith(img_name)]
        
        if matching_masks:
            pairs.append((img_file, matching_masks))
        else:
            orphaned_images.append(img_file)
    
    # Find orphaned masks
    used_masks = set()
    for _, masks in pairs:
        used_masks.update(masks)
    
    orphaned_masks = [m for m in mask_files if m not in used_masks]
    
    stats = {
        'total_images': len(image_files),
        'total_masks': len(mask_files),
        'matched_pairs': len(pairs),
        'orphaned_images': len(orphaned_images),
        'orphaned_masks': len(orphaned_masks),
        'multiple_masks_per_image': sum(1 for _, masks in pairs if len(masks) > 1)
    }
    
    return stats


def visualize_sample_data(img_dir: str, mask_dir: str, num_samples: int = 6):
    """Visualize sample image-mask pairs"""
    
    print(f"Visualizing {num_samples} sample pairs...")
    
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
    
    mask_files = [f for f in os.listdir(mask_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Find pairs
    pairs = []
    for img_name, img_file in image_files.items():
        matching_masks = [m for m in mask_files if m.startswith(img_name)]
        if matching_masks:
            pairs.append((img_file, matching_masks[0]))  # Take first mask
    
    if not pairs:
        print("No matching pairs found!")
        return
    
    # Sample random pairs
    sample_pairs = np.random.choice(len(pairs), min(num_samples, len(pairs)), replace=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, idx in enumerate(sample_pairs):
        img_file, mask_file = pairs[idx]
        
        # Load image
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Display
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image: {img_file}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f"Mask: {mask_file}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_statistics(img_stats: Dict, mask_stats: Dict):
    """Plot dataset statistics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Image dimensions
    if img_stats.get('aspect_ratios'):
        axes[0, 0].hist(img_stats['aspect_ratios'], bins=20, alpha=0.7)
        axes[0, 0].set_title('Image Aspect Ratios')
        axes[0, 0].set_xlabel('Width/Height')
        axes[0, 0].set_ylabel('Count')
    
    # File sizes
    if img_stats.get('file_size_stats'):
        # Create dummy data for visualization since we don't have the raw values
        axes[0, 1].bar(['Min', 'Mean', 'Max'], 
                      [img_stats['file_size_stats']['min'],
                       img_stats['file_size_stats']['mean'],
                       img_stats['file_size_stats']['max']])
        axes[0, 1].set_title('Image File Sizes (KB)')
        axes[0, 1].set_ylabel('Size (KB)')
    
    # Channel distribution
    if img_stats.get('channels'):
        channels = list(img_stats['channels'].keys())
        counts = list(img_stats['channels'].values())
        axes[0, 2].bar(channels, counts)
        axes[0, 2].set_title('Channel Distribution')
        axes[0, 2].set_xlabel('Number of Channels')
        axes[0, 2].set_ylabel('Count')
    
    # Mask coverage
    if mask_stats:
        axes[1, 0].bar(['Min', 'Mean', 'Max'], 
                      [mask_stats['coverage_stats']['min'],
                       mask_stats['coverage_stats']['mean'],
                       mask_stats['coverage_stats']['max']])
        axes[1, 0].set_title('Mask Coverage (%)')
        axes[1, 0].set_ylabel('Coverage %')
    
    # Mask values
    if mask_stats.get('value_counts'):
        values = list(mask_stats['value_counts'].keys())[:10]  # Top 10
        counts = [mask_stats['value_counts'][v] for v in values]
        axes[1, 1].bar(values, counts)
        axes[1, 1].set_title('Mask Pixel Values')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Count')
    
    # Summary text
    summary_text = f"""
Dataset Summary:
Images: {img_stats.get('count', 0)}
Masks: {mask_stats.get('count', 0)}

Avg Image Size: {img_stats['width_stats']['mean']:.0f}x{img_stats['height_stats']['mean']:.0f}
Avg Mask Coverage: {mask_stats['coverage_stats']['mean']:.1f}%
"""
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def recommend_config(img_stats: Dict, mask_stats: Dict) -> Dict:
    """Recommend configuration based on data analysis"""
    
    recommendations = {}
    
    # Image size recommendation
    avg_width = img_stats['width_stats']['mean']
    avg_height = img_stats['height_stats']['mean']
    
    # Suggest power of 2 sizes close to average
    size_options = [256, 384, 512, 640, 768, 1024]
    target_size = max(avg_width, avg_height)
    recommended_size = min(size_options, key=lambda x: abs(x - target_size))
    
    recommendations['img_size'] = (recommended_size, recommended_size)
    
    # Batch size recommendation based on image size
    if recommended_size <= 256:
        recommendations['batch_size'] = 16
    elif recommended_size <= 512:
        recommendations['batch_size'] = 8
    elif recommended_size <= 768:
        recommendations['batch_size'] = 4
    else:
        recommendations['batch_size'] = 2
    
    # Model recommendation based on dataset size
    num_images = img_stats.get('count', 0)
    if num_images < 100:
        recommendations['model'] = 'Unet'
        recommendations['encoder'] = 'resnet34'
    elif num_images < 500:
        recommendations['model'] = 'UnetPlusPlus'
        recommendations['encoder'] = 'efficientnet-b2'
    else:
        recommendations['model'] = 'UnetPlusPlus'
        recommendations['encoder'] = 'efficientnet-b4'
    
    # Loss function recommendation based on mask coverage
    avg_coverage = mask_stats['coverage_stats']['mean']
    if avg_coverage < 5:  # Very sparse masks
        recommendations['loss'] = 'focal'
    elif avg_coverage < 20:  # Moderately sparse
        recommendations['loss'] = 'dice_focal'
    else:  # Balanced
        recommendations['loss'] = 'dice'
    
    return recommendations


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Explore worm segmentation dataset')
    parser.add_argument('--data-root', type=str, 
                       default="/mnt/c/Users/samuel/OneDrive/Documents/worms/zoomwormtracking/supervisely_download",
                       help='Root directory containing images and masks')
    parser.add_argument('--img-dir', type=str, default='images',
                       help='Image directory name')
    parser.add_argument('--mask-dir', type=str, default='masks_png',
                       help='Mask directory name')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualizations')
    
    args = parser.parse_args()
    
    # Setup paths
    img_path = os.path.join(args.data_root, args.img_dir)
    mask_path = os.path.join(args.data_root, args.mask_dir)
    
    print("🔍 Exploring Worm Segmentation Dataset")
    print("=" * 50)
    
    # Check if paths exist
    if not os.path.exists(img_path):
        print(f"❌ Image directory not found: {img_path}")
        return
    
    if not os.path.exists(mask_path):
        print(f"❌ Mask directory not found: {mask_path}")
        return
    
    # Analyze images
    print("\n📸 Analyzing Images...")
    img_stats = analyze_images(img_path)
    
    if img_stats:
        print(f"   Total images: {img_stats['count']}")
        print(f"   Average size: {img_stats['width_stats']['mean']:.0f}x{img_stats['height_stats']['mean']:.0f}")
        print(f"   Size range: {img_stats['width_stats']['min']:.0f}-{img_stats['width_stats']['max']:.0f} x {img_stats['height_stats']['min']:.0f}-{img_stats['height_stats']['max']:.0f}")
        print(f"   Average file size: {img_stats['file_size_stats']['mean']:.1f} KB")
    
    # Analyze masks
    print("\n🎭 Analyzing Masks...")
    mask_stats = analyze_masks(mask_path)
    
    if mask_stats:
        print(f"   Total masks: {mask_stats['count']}")
        print(f"   Average coverage: {mask_stats['coverage_stats']['mean']:.1f}%")
        print(f"   Coverage range: {mask_stats['coverage_stats']['min']:.1f}% - {mask_stats['coverage_stats']['max']:.1f}%")
        print(f"   Unique values: {mask_stats['unique_values']}")
    
    # Check pairs
    print("\n🔗 Checking Image-Mask Pairs...")
    pair_stats = check_image_mask_pairs(img_path, mask_path)
    
    print(f"   Matched pairs: {pair_stats['matched_pairs']}")
    print(f"   Orphaned images: {pair_stats['orphaned_images']}")
    print(f"   Orphaned masks: {pair_stats['orphaned_masks']}")
    print(f"   Images with multiple masks: {pair_stats['multiple_masks_per_image']}")
    
    # Generate recommendations
    if img_stats and mask_stats:
        print("\n💡 Configuration Recommendations...")
        recommendations = recommend_config(img_stats, mask_stats)
        
        print(f"   Recommended image size: {recommendations['img_size']}")
        print(f"   Recommended batch size: {recommendations['batch_size']}")
        print(f"   Recommended model: {recommendations['model']}")
        print(f"   Recommended encoder: {recommendations['encoder']}")
        print(f"   Recommended loss: {recommendations['loss']}")
    
    # Visualizations
    if args.visualize and img_stats and mask_stats:
        print("\n📊 Creating Visualizations...")
        
        # Plot statistics
        plot_statistics(img_stats, mask_stats)
        
        # Show sample data
        visualize_sample_data(img_path, mask_path, num_samples=6)
    
    print("\n✅ Data exploration completed!")


if __name__ == "__main__":
    main() 
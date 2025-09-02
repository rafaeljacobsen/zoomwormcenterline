#!/usr/bin/env python3
"""
Detect curled/overlapping worms by analyzing mask thickness
Usage: python detect_curled_worms.py --input video_results --thickness_threshold 2.0
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt


def calculate_mask_thickness(mask):
    """Calculate average thickness of a mask using distance transform"""
    if np.sum(mask) == 0:
        return 0
    
    # Convert to binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Distance transform - gives distance to nearest boundary
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Average thickness is 2x the mean distance (diameter)
    mean_distance = np.mean(dist_transform[dist_transform > 0])
    thickness = 2 * mean_distance
    
    return thickness


def calculate_skeleton_thickness(mask):
    """Alternative method using skeletonization"""
    if np.sum(mask) == 0:
        return 0
    
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Skeletonize to get centerline
    skeleton = cv2.ximgproc.thinning(binary_mask)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Get distances along skeleton
    skeleton_distances = dist_transform[skeleton > 0]
    
    if len(skeleton_distances) == 0:
        return 0
    
    # Average thickness along skeleton
    return 2 * np.mean(skeleton_distances)


def analyze_worm_masks(input_folder):
    """Analyze all masks to find thickness statistics"""
    
    mask_files = sorted([f for f in os.listdir(input_folder) if f.endswith('_mask.png')])
    
    if not mask_files:
        print("No mask files found!")
        return None, None
    
    print(f"Analyzing {len(mask_files)} masks...")
    
    thicknesses = []
    frame_data = []
    
    for mask_file in tqdm(mask_files, desc="Computing thickness"):
        mask_path = os.path.join(input_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is not None:
            thickness = calculate_mask_thickness(mask)
            thicknesses.append(thickness)
            
            # Extract frame number
            frame_num = int(mask_file.split('_')[1])
            frame_data.append((frame_num, thickness, mask_file))
    
    thicknesses = np.array(thicknesses)
    
    # Calculate statistics
    mean_thickness = np.mean(thicknesses)
    std_thickness = np.std(thicknesses)
    median_thickness = np.median(thicknesses)
    
    print(f"\nThickness Statistics:")
    print(f"  Mean: {mean_thickness:.2f}")
    print(f"  Median: {median_thickness:.2f}")
    print(f"  Std: {std_thickness:.2f}")
    print(f"  Range: {np.min(thicknesses):.2f} - {np.max(thicknesses):.2f}")
    
    return frame_data, {
        'mean': mean_thickness,
        'std': std_thickness,
        'median': median_thickness,
        'min': np.min(thicknesses),
        'max': np.max(thicknesses)
    }


def detect_thick_frames(frame_data, stats, thickness_threshold=2.0):
    """Detect frames where worm is unusually thick (curled)"""
    
    threshold_value = stats['mean'] + thickness_threshold * stats['std']
    
    thick_frames = []
    for frame_num, thickness, mask_file in frame_data:
        if thickness > threshold_value:
            thick_frames.append((frame_num, thickness, mask_file))
    
    print(f"\nThickness threshold: {threshold_value:.2f}")
    print(f"Found {len(thick_frames)} frames with thick worms")
    
    return thick_frames, threshold_value


def export_curled_frames(input_folder, thick_frames, video_path):
    """Export original frames of curled worms to a separate folder"""
    
    if not thick_frames:
        print("No thick frames to export")
        return
    
    # Create output folder
    output_folder = os.path.join(input_folder, "curled_worms")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Exporting {len(thick_frames)} original frames to {output_folder}...")
    
    # Open video to extract original frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    import shutil
    
    for frame_num, thickness, mask_file in tqdm(thick_frames, desc="Extracting frames"):
        # Set video position to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Save original frame
            frame_filename = f"frame_{frame_num:06d}_original.png"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
    
    cap.release()
    print(f"✅ Exported {len(thick_frames)} original frames to {output_folder}")


def create_curled_worm_video(input_folder, thick_frames, output_name="curled_worms"):
    """Create video of frames with curled worms"""
    
    if not thick_frames:
        print("No thick frames to create video")
        return
    
    print(f"Creating video with {len(thick_frames)} frames...")
    
    # Find corresponding overlay files
    overlay_frames = []
    for frame_num, thickness, mask_file in thick_frames:
        overlay_file = mask_file.replace('_mask.png', '_overlay.png')
        overlay_path = os.path.join(input_folder, overlay_file)
        
        if os.path.exists(overlay_path):
            overlay_frames.append((frame_num, overlay_path, thickness))
    
    if not overlay_frames:
        print("No overlay files found for thick frames")
        return
    
    # Get frame size from first overlay
    first_overlay = cv2.imread(overlay_frames[0][1])
    height, width = first_overlay.shape[:2]
    
    # Try different codecs
    codecs = [('XVID', '.avi'), ('MJPG', '.avi'), ('mp4v', '.mp4')]
    
    for codec, ext in codecs:
        try:
            output_path = os.path.join(input_folder, f'{output_name}{ext}')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height), isColor=True)
            
            if writer.isOpened():
                print(f"Creating video with {codec} codec...")
                
                for frame_num, overlay_path, thickness in tqdm(overlay_frames, desc="Writing frames"):
                    overlay = cv2.imread(overlay_path)
                    
                    if overlay is not None:
                        # Add thickness text to frame
                        text = f"Frame {frame_num}, Thickness: {thickness:.1f}"
                        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 255, 0), 2)
                        writer.write(overlay)
                
                writer.release()
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    print(f"✅ Created: {output_path}")
                    return
                
        except Exception as e:
            print(f"Failed with {codec}: {e}")
    
    print("❌ Failed to create video")


def plot_thickness_distribution(frame_data, stats, thick_frames, threshold_value):
    """Plot thickness distribution"""
    
    thicknesses = [thickness for _, thickness, _ in frame_data]
    thick_thicknesses = [thickness for _, thickness, _ in thick_frames]
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(thicknesses, bins=50, alpha=0.7, label='All frames')
    plt.hist(thick_thicknesses, bins=50, alpha=0.7, label='Thick frames', color='red')
    plt.axvline(stats['mean'], color='blue', linestyle='--', label=f"Mean: {stats['mean']:.1f}")
    plt.axvline(threshold_value, color='red', linestyle='--', label=f"Threshold: {threshold_value:.1f}")
    plt.xlabel('Thickness')
    plt.ylabel('Count')
    plt.title('Worm Thickness Distribution')
    plt.legend()
    
    # Time series
    plt.subplot(1, 2, 2)
    frame_nums = [frame_num for frame_num, _, _ in frame_data]
    plt.plot(frame_nums, thicknesses, 'b-', alpha=0.7, label='Thickness')
    
    thick_frame_nums = [frame_num for frame_num, _, _ in thick_frames]
    thick_vals = [thickness for _, thickness, _ in thick_frames]
    plt.scatter(thick_frame_nums, thick_vals, color='red', s=20, label='Thick frames')
    
    plt.axhline(threshold_value, color='red', linestyle='--', label=f"Threshold: {threshold_value:.1f}")
    plt.xlabel('Frame Number')
    plt.ylabel('Thickness')
    plt.title('Thickness Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(frame_data[0][2]), 'thickness_analysis.png'), dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Detect curled worms by thickness analysis')
    parser.add_argument('--input', type=str, default='video_results', 
                       help='Input folder with mask and overlay files')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to original video file')
    parser.add_argument('--thickness_threshold', type=float, default=2.0,
                       help='Standard deviations above mean for thick detection')
    parser.add_argument('--plot', action='store_true',
                       help='Show thickness distribution plot')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input folder not found: {args.input}")
        return
    
    # Analyze masks
    frame_data, stats = analyze_worm_masks(args.input)
    
    if frame_data is None:
        return
    
    # Detect thick frames
    thick_frames, threshold_value = detect_thick_frames(frame_data, stats, args.thickness_threshold)
    
    # Export curled frames to folder
    export_curled_frames(args.input, thick_frames, args.video)
    
    # Create video of curled worms
    create_curled_worm_video(args.input, thick_frames)
    
    # Plot if requested
    if args.plot:
        plot_thickness_distribution(frame_data, stats, thick_frames, threshold_value)
    
    print(f"\nSummary:")
    print(f"  Total frames: {len(frame_data)}")
    print(f"  Thick frames: {len(thick_frames)} ({100*len(thick_frames)/len(frame_data):.1f}%)")
    print(f"  Thickness threshold: {threshold_value:.2f}")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Create videos from processed frames
Usage: python create_videos.py --input video_results --fps 30
"""

import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np


def create_videos_from_frames(input_folder: str, fps: float = 30.0):
    """Create videos from mask and overlay frames"""
    
    print(f"Creating videos from frames in: {input_folder}")
    
    # Find frame files
    mask_files = sorted([f for f in os.listdir(input_folder) if f.endswith('_mask.png')])
    overlay_files = sorted([f for f in os.listdir(input_folder) if f.endswith('_overlay.png')])
    
    if not mask_files:
        print("❌ No mask files found!")
        return
    
    print(f"Found {len(mask_files)} mask frames")
    print(f"Found {len(overlay_files)} overlay frames")
    
    # Get frame size from first mask
    first_mask = cv2.imread(os.path.join(input_folder, mask_files[0]), cv2.IMREAD_GRAYSCALE)
    height, width = first_mask.shape
    frame_size = (width, height)
    
    print(f"Frame size: {frame_size}")
    
    # Try different codecs
    codecs_to_try = [
        ('XVID', '.avi'),
        ('MJPG', '.avi'),
        ('mp4v', '.mp4'),
    ]
    
    # Create mask video
    mask_success = False
    for codec, ext in codecs_to_try:
        try:
            output_path = os.path.join(input_folder, f'masks_video{ext}')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size, isColor=False)
            
            if writer.isOpened():
                print(f"Creating mask video with {codec} codec...")
                
                for mask_file in tqdm(mask_files, desc="Processing masks"):
                    mask_path = os.path.join(input_folder, mask_file)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    if mask is not None:
                        # Ensure correct size
                        if mask.shape != (height, width):
                            mask = cv2.resize(mask, frame_size)
                        writer.write(mask)
                
                writer.release()
                
                # Check if file was actually created and has size
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    print(f"✅ Successfully created: {output_path}")
                    mask_success = True
                    break
                else:
                    print(f"❌ File created but appears empty: {output_path}")
                    if os.path.exists(output_path):
                        os.remove(output_path)
        
        except Exception as e:
            print(f"❌ Failed with {codec}: {e}")
            continue
    
    if not mask_success:
        print("❌ Failed to create mask video with any codec")
    
    # Create overlay video if overlay frames exist
    if overlay_files:
        overlay_success = False
        
        # Get frame size from first overlay
        first_overlay = cv2.imread(os.path.join(input_folder, overlay_files[0]))
        if first_overlay is not None:
            overlay_height, overlay_width = first_overlay.shape[:2]
            overlay_frame_size = (overlay_width, overlay_height)
            
            for codec, ext in codecs_to_try:
                try:
                    output_path = os.path.join(input_folder, f'overlay_video{ext}')
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(output_path, fourcc, fps, overlay_frame_size, isColor=True)
                    
                    if writer.isOpened():
                        print(f"Creating overlay video with {codec} codec...")
                        
                        for overlay_file in tqdm(overlay_files, desc="Processing overlays"):
                            overlay_path = os.path.join(input_folder, overlay_file)
                            overlay = cv2.imread(overlay_path)
                            
                            if overlay is not None:
                                # Ensure correct size
                                if overlay.shape[:2] != (overlay_height, overlay_width):
                                    overlay = cv2.resize(overlay, overlay_frame_size)
                                writer.write(overlay)
                        
                        writer.release()
                        
                        # Check if file was actually created and has size
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                            print(f"✅ Successfully created: {output_path}")
                            overlay_success = True
                            break
                        else:
                            print(f"❌ File created but appears empty: {output_path}")
                            if os.path.exists(output_path):
                                os.remove(output_path)
                
                except Exception as e:
                    print(f"❌ Failed with {codec}: {e}")
                    continue
        
        if not overlay_success:
            print("❌ Failed to create overlay video")
    
    # Summary
    if mask_success or overlay_success:
        print("\n🎉 Video creation completed!")
        print("Check your output folder for the created videos.")
    else:
        print("\n❌ No videos could be created")
        print("The individual frame images are still available.")


def main():
    parser = argparse.ArgumentParser(description='Create videos from processed frames')
    parser.add_argument('--input', type=str, default='video_results', 
                       help='Input folder containing frame images')
    parser.add_argument('--fps', type=float, default=30.0, 
                       help='Frames per second for output video')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input folder not found: {args.input}")
        return
    
    create_videos_from_frames(args.input, args.fps)


if __name__ == "__main__":
    main() 
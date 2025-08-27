#!/usr/bin/env python3
"""
Fast video inference for worm segmentation
Processes MP4 videos frame by frame with batch processing
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import argparse

from config import Config
from inference import SegmentationInference


def process_video_fast(
    video_path: str,
    model_path: str,
    config: Config,
    output_folder: str,
    batch_size: int = 32,
    skip_frames: int = 1,
    save_overlays: bool = True,
    apply_postprocessing: bool = True
):
    """
    Process video frames in batches for maximum speed
    
    Args:
        video_path: Path to input MP4 video
        model_path: Path to trained model checkpoint
        config: Configuration object
        output_folder: Where to save results
        batch_size: Number of frames to process at once
        skip_frames: Process every Nth frame (1 = all frames, 2 = every other frame)
        save_overlays: Whether to save overlay visualizations
        apply_postprocessing: Whether to apply morphological operations
    """
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize inference
    print("Loading model...")
    inference = SegmentationInference(model_path, config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
    print(f"Processing every {skip_frames} frame(s) in batches of {batch_size}")
    
    # Process frames in batches
    frame_batch = []
    frame_indices = []
    processed_count = 0
    
    with tqdm(total=total_frames//skip_frames, desc="Processing frames") as pbar:
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % skip_frames == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_batch.append(frame_rgb)
                frame_indices.append(frame_idx)
                
                # Process batch when full
                if len(frame_batch) == batch_size:
                    process_frame_batch(
                        frame_batch, frame_indices, inference, output_folder,
                        save_overlays, apply_postprocessing
                    )
                    processed_count += len(frame_batch)
                    pbar.update(len(frame_batch))
                    
                    # Clear batch
                    frame_batch = []
                    frame_indices = []
            
            frame_idx += 1
        
        # Process remaining frames
        if frame_batch:
            process_frame_batch(
                frame_batch, frame_indices, inference, output_folder,
                save_overlays, apply_postprocessing
            )
            processed_count += len(frame_batch)
            pbar.update(len(frame_batch))
    
    cap.release()
    print(f"Processed {processed_count} frames successfully!")
    
    # Create video from masks if requested
    create_result_video(output_folder, fps/skip_frames, config.img_size)


def process_frame_batch(
    frames, frame_indices, inference, output_folder, 
    save_overlays, apply_postprocessing
):
    """Process a batch of frames"""
    
    # Preprocess all frames
    frame_tensors = []
    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, inference.config.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply transforms
        transformed = inference.transform(image=resized)
        frame_tensor = transformed['image'].unsqueeze(0)
        frame_tensors.append(frame_tensor)
    
    # Stack into batch
    batch_tensor = torch.cat(frame_tensors, dim=0).to(inference.device)
    
    # Run inference on batch
    with torch.no_grad():
        outputs = inference.model(batch_tensor)
    
    # Process outputs
    for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
        # Get mask
        mask = inference.postprocess_mask(outputs[i:i+1], threshold=0.5)
        
        # Apply post-processing
        if apply_postprocessing:
            mask = inference.apply_morphological_operations(mask, 'both')
            mask = inference.filter_by_area(mask, min_area=100)
        
        # Save mask
        mask_path = os.path.join(output_folder, f"frame_{frame_idx:06d}_mask.png")
        cv2.imwrite(mask_path, mask)
        
        # Save overlay if requested
        if save_overlays:
            overlay = inference.create_contour_overlay(frame, mask)
            overlay_path = os.path.join(output_folder, f"frame_{frame_idx:06d}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def create_result_video(output_folder: str, fps: float, img_size: tuple):
    """Create MP4 videos from processed frames"""
    
    # Find mask files
    mask_files = sorted([f for f in os.listdir(output_folder) if f.endswith('_mask.png')])
    overlay_files = sorted([f for f in os.listdir(output_folder) if f.endswith('_overlay.png')])
    
    if not mask_files:
        print("No mask files found to create video")
        return
    
    # Create mask video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_writer = cv2.VideoWriter(
        os.path.join(output_folder, 'masks_video.mp4'),
        fourcc, fps, img_size, isColor=False
    )
    
    print("Creating mask video...")
    for mask_file in tqdm(mask_files):
        mask = cv2.imread(os.path.join(output_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        mask_writer.write(mask)
    mask_writer.release()
    
    # Create overlay video if overlays exist
    if overlay_files:
        overlay_writer = cv2.VideoWriter(
            os.path.join(output_folder, 'overlay_video.mp4'),
            fourcc, fps, img_size, isColor=True
        )
        
        print("Creating overlay video...")
        for overlay_file in tqdm(overlay_files):
            overlay = cv2.imread(os.path.join(output_folder, overlay_file))
            overlay_writer.write(overlay)
        overlay_writer.release()
    
    print("Videos created successfully!")


def main():
    parser = argparse.ArgumentParser(description='Fast video inference for worm segmentation')
    parser.add_argument('--video', type=str, required=True, help='Input MP4 video path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='video_results', help='Output folder')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--no-overlays', action='store_true', help='Skip overlay generation')
    parser.add_argument('--no-postprocessing', action='store_true', help='Skip post-processing')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Process video
    process_video_fast(
        video_path=args.video,
        model_path=args.checkpoint,
        config=config,
        output_folder=args.output,
        batch_size=args.batch_size,
        skip_frames=args.skip_frames,
        save_overlays=not args.no_overlays,
        apply_postprocessing=not args.no_postprocessing
    )


if __name__ == "__main__":
    main() 
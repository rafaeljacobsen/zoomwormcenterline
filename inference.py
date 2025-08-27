import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from trainer import load_trained_model
from dataset import get_transforms


class SegmentationInference:
    """Class for running inference on segmentation models"""
    
    def __init__(self, model_path: str, config: Config, device: str = 'auto'):
        self.config = config
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = load_trained_model(model_path, config)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_transforms(config, is_train=False)
        
        print(f"Model loaded on {self.device}")
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        """Preprocess a single image for inference"""
        
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, self.config.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def postprocess_mask(self, mask: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Postprocess model output to create final mask"""
        
        # Apply sigmoid and threshold
        mask = torch.sigmoid(mask)
        mask = (mask > threshold).float()
        
        # Convert to numpy
        mask = mask.cpu().squeeze().numpy()
        
        # Convert to uint8
        mask = (mask * 255).astype(np.uint8)
        
        return mask
    
    def predict_single(
        self, 
        image: Union[str, np.ndarray], 
        threshold: float = 0.5,
        use_tta: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mask for a single image"""
        
        # Preprocess
        image_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            if use_tta and self.config.test_time_augmentation:
                # Test-time augmentation
                outputs = self.predict_with_tta(image_tensor)
            else:
                outputs = self.model(image_tensor)
        
        # Postprocess
        mask = self.postprocess_mask(outputs, threshold)
        
        # Get original image for visualization
        if isinstance(image, str):
            orig_image = cv2.imread(image)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        else:
            orig_image = image.copy()
        
        return orig_image, mask
    
    def predict_with_tta(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Test-time augmentation for better predictions"""
        
        # Original prediction
        outputs = [self.model(image_tensor)]
        
        # Horizontal flip
        flipped_h = torch.flip(image_tensor, dims=[3])
        pred_h = self.model(flipped_h)
        pred_h = torch.flip(pred_h, dims=[3])
        outputs.append(pred_h)
        
        # Vertical flip
        flipped_v = torch.flip(image_tensor, dims=[2])
        pred_v = self.model(flipped_v)
        pred_v = torch.flip(pred_v, dims=[2])
        outputs.append(pred_v)
        
        # Both flips
        flipped_hv = torch.flip(image_tensor, dims=[2, 3])
        pred_hv = self.model(flipped_hv)
        pred_hv = torch.flip(pred_hv, dims=[2, 3])
        outputs.append(pred_hv)
        
        # Average all predictions
        return torch.mean(torch.stack(outputs), dim=0)
    
    def predict_batch(
        self, 
        image_paths: List[str], 
        threshold: float = 0.5,
        batch_size: int = 8
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Predict masks for a batch of images"""
        
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for path in batch_paths:
                tensor = self.preprocess_image(path)
                batch_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            # Postprocess each image in batch
            for j, path in enumerate(batch_paths):
                mask = self.postprocess_mask(outputs[j:j+1], threshold)
                
                # Load original image
                orig_image = cv2.imread(path)
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                
                results.append((orig_image, mask))
        
        return results
    
    def visualize_prediction(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        title: str = "Segmentation Result",
        save_path: Optional[str] = None
    ):
        """Visualize segmentation result"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Predicted Mask")
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 0] = mask  # Red channel for mask
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_contour_overlay(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        contour_color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Create an overlay with mask contours"""
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on image
        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, contour_color, thickness)
        
        return overlay
    
    def apply_morphological_operations(
        self, 
        mask: np.ndarray,
        operation: str = 'close',
        kernel_size: int = 5
    ) -> np.ndarray:
        """Apply morphological operations to clean up mask"""
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'close':
            # Fill small holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            # Remove small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'both':
            # Apply both operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def filter_by_area(
        self, 
        mask: np.ndarray, 
        min_area: int = 100, 
        max_area: Optional[int] = None
    ) -> np.ndarray:
        """Filter mask components by area"""
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Create filtered mask
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Check area constraints
            if area >= min_area and (max_area is None or area <= max_area):
                filtered_mask[labels == i] = 255
        
        return filtered_mask


def run_inference_on_folder(
    model_path: str,
    config: Config,
    input_folder: str,
    output_folder: str,
    threshold: float = 0.5,
    use_tta: bool = False,
    apply_postprocessing: bool = True
):
    """Run inference on all images in a folder"""
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize inference
    inference = SegmentationInference(model_path, config)
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_folder, file))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images in batches for speed
    batch_size = 16  # Process 16 images at once
    print(f"Processing {len(image_files)} images in batches of {batch_size}...")
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}: images {i+1}-{min(i+batch_size, len(image_files))}")
        
        try:
            # Predict batch
            results = inference.predict_batch(batch_files, threshold, batch_size)
            
            # Save results for each image in batch
            for j, (image_path, (image, mask)) in enumerate(zip(batch_files, results)):
            
                # Apply post-processing if requested
                if apply_postprocessing:
                    mask = inference.apply_morphological_operations(mask, 'both')
                    mask = inference.filter_by_area(mask, min_area=100)
                
                # Save results
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save mask
                mask_path = os.path.join(output_folder, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, mask)
                
                # Save overlay
                overlay = inference.create_contour_overlay(image, mask)
                overlay_path = os.path.join(output_folder, f"{base_name}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
    
    print("Inference completed!")


if __name__ == "__main__":
    from config import Config
    
    # Example usage
    config = Config()
    model_path = "checkpoints/best-epoch=50-val_dice=0.9500.ckpt"
    
    # Run inference on a single image
    inference = SegmentationInference(model_path, config)
    image_path = "test_image.jpg"
    
    if os.path.exists(image_path):
        image, mask = inference.predict_single(image_path, threshold=0.5, use_tta=True)
        inference.visualize_prediction(image, mask, "Test Prediction") 
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class WormSegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths: List[str], 
        mask_paths: List[str], 
        transform: Optional[A.Compose] = None,
        img_size: Tuple[int, int] = (512, 512)
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.img_size = img_size
        
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize to target size
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask to 0-1
        mask = mask.astype(np.float32) / 255.0
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)  # Add channel dimension to mask


def get_transforms(config, is_train: bool = True) -> A.Compose:
    """Create augmentation transforms for training and validation"""
    
    if is_train and config.use_augmentation:
        transform = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=45, 
                p=0.5
            ),
            
            # Color and intensity augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            
            # Normalize and convert to tensor
            A.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    return transform


def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Get all image names (without extension)
    image_files = [f for f in os.listdir(config.img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_names = [os.path.splitext(f)[0] for f in image_files]
    
    # Find corresponding masks for each image
    image_paths = []
    mask_paths = []
    
    for name in image_names:
        # Try different image extensions
        img_path = None
        for ext in ['.bmp', '.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(config.img_path, f"{name}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        # Find corresponding mask(s)
        mask_files = [f for f in os.listdir(config.mask_path) if f.startswith(name)]
        
        if mask_files and os.path.exists(img_path):
            # Take the first mask if multiple exist
            mask_path = os.path.join(config.mask_path, sorted(mask_files)[0])
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    
    print(f"Found {len(image_paths)} image-mask pairs")
    
    # Split data
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, 
        test_size=(1 - config.train_split), 
        random_state=42
    )
    
    val_size = config.val_split / (config.val_split + config.test_split)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, 
        test_size=(1 - val_size), 
        random_state=42
    )
    
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    # Create datasets
    train_dataset = WormSegmentationDataset(
        train_imgs, train_masks, 
        transform=get_transforms(config, is_train=True),
        img_size=config.img_size
    )
    
    val_dataset = WormSegmentationDataset(
        val_imgs, val_masks, 
        transform=get_transforms(config, is_train=False),
        img_size=config.img_size
    )
    
    test_dataset = WormSegmentationDataset(
        test_imgs, test_masks, 
        transform=get_transforms(config, is_train=False),
        img_size=config.img_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader


def visualize_batch(dataloader: DataLoader, config, num_samples: int = 4):
    """Visualize a batch of images and masks"""
    batch = next(iter(dataloader))
    images, masks = batch
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(min(num_samples, images.shape[0])):
        # Denormalize image for visualization
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array(config.normalize_std) + np.array(config.normalize_mean)
        img = np.clip(img, 0, 1)
        
        mask = masks[i].squeeze().numpy()
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f"Mask {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show() 
import os
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Config:
    # Data paths
    data_root: str = "/mnt/c/Users/samuel/OneDrive/Documents/worms/zoomwormtracking/supervisely_download"
    img_dir: str = "images"
    mask_dir: str = "masks_png"
    
    # Model architecture
    model_name: str = "UnetPlusPlus"  # Options: "UnetPlusPlus", "DeepLabV3Plus", "Unet"
    encoder_name: str = "efficientnet-b4"  # Backbone encoder
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 1  # Binary segmentation (assuming single class masks)
    
    # Training parameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Image processing
    img_size: Tuple[int, int] = (512, 512)  # Resize images to this size
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet stats
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Training settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Loss and optimization
    loss_function: str = "dice_focal"  # Options: "dice", "focal", "dice_focal", "bce"
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Monitoring and logging
    use_wandb: bool = True
    project_name: str = "worm-segmentation"
    save_dir: str = "checkpoints"
    log_every_n_steps: int = 10
    
    # Inference
    test_time_augmentation: bool = True
    tta_transforms: int = 4
    
    # Model ensemble (if using multiple models)
    use_ensemble: bool = False
    ensemble_models: List[str] = None
    
    def __post_init__(self):
        # Create absolute paths
        self.img_path = os.path.join(self.data_root, self.img_dir)
        self.mask_path = os.path.join(self.data_root, self.mask_dir)
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True) 
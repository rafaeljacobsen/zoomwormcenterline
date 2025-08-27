import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, List
import timm


class SegmentationModel(nn.Module):
    """Wrapper for various segmentation models"""
    
    def __init__(
        self,
        model_name: str = "UnetPlusPlus",
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.classes = classes
        
        # Create the model based on the specified architecture
        if model_name == "UnetPlusPlus":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        elif model_name == "DeepLabV3Plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        elif model_name == "Unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        elif model_name == "FPN":
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        elif model_name == "PSPNet":
            self.model = smp.PSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        elif model_name == "MAnet":
            self.model = smp.MAnet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EnsembleModel(nn.Module):
    """Ensemble of multiple segmentation models for improved performance"""
    
    def __init__(self, models: List[SegmentationModel], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average of predictions
        ensemble_output = sum(w * out for w, out in zip(self.weights, outputs))
        return ensemble_output


class MultiScaleModel(nn.Module):
    """Multi-scale prediction model for better boundary detection"""
    
    def __init__(
        self,
        base_model: SegmentationModel,
        scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]
    ):
        super().__init__()
        self.base_model = base_model
        self.scales = scales
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # Process at multiple scales
        outputs = []
        for scale in self.scales:
            if scale != 1.0:
                # Resize input
                new_h, new_w = int(height * scale), int(width * scale)
                scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                # Forward pass
                scaled_output = self.base_model(scaled_x)
                
                # Resize back to original size
                output = F.interpolate(scaled_output, size=(height, width), mode='bilinear', align_corners=False)
            else:
                output = self.base_model(x)
            
            outputs.append(output)
        
        # Average predictions across scales
        return torch.mean(torch.stack(outputs), dim=0)


def create_model(config) -> nn.Module:
    """Factory function to create segmentation models based on config"""
    
    model = SegmentationModel(
        model_name=config.model_name,
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        classes=config.classes,
        activation=None  # We'll apply sigmoid/softmax in the loss function
    )
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """Get information about the model architecture"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
    } 
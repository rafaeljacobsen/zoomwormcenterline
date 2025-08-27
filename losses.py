import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    
    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid if predictions are logits
        if not torch.all((predictions >= 0) & (predictions <= 1)):
            predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Remove ignored indices
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Calculate dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Use binary_cross_entropy_with_logits for mixed precision safety
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Apply sigmoid for focal weight calculation
        predictions = torch.sigmoid(predictions)
        
        # Compute focal weight
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice loss"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid if predictions are logits
        if not torch.all((predictions >= 0) & (predictions <= 1)):
            predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate True Positives, False Positives, False Negatives
        tp = (predictions * targets).sum()
        fp = (predictions * (1 - targets)).sum()
        fn = ((1 - predictions) * targets).sum()
        
        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky


class IoULoss(nn.Module):
    """Intersection over Union (IoU) Loss"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid if predictions are logits
        if not torch.all((predictions >= 0) & (predictions <= 1)):
            predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class CombinedLoss(nn.Module):
    """Combination of multiple loss functions"""
    
    def __init__(
        self,
        losses: dict,
        weights: Optional[dict] = None
    ):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        
        if weights is None:
            self.weights = {name: 1.0 for name in losses.keys()}
        else:
            self.weights = weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        
        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            loss_value = loss_fn(predictions, targets)
            total_loss += weight * loss_value
        
        return total_loss


class DiceFocalLoss(nn.Module):
    """Combined Dice and Focal Loss"""
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1e-6
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_loss(predictions, targets)
        focal_loss = self.focal_loss(predictions, targets)
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


def get_loss_function(config):
    """Factory function to create loss functions based on config"""
    
    if config.loss_function == "dice":
        return DiceLoss()
    elif config.loss_function == "focal":
        return FocalLoss(alpha=1.0, gamma=2.0)
    elif config.loss_function == "dice_focal":
        return DiceFocalLoss(dice_weight=0.5, focal_weight=0.5)
    elif config.loss_function == "tversky":
        return TverskyLoss(alpha=0.3, beta=0.7)  # Focus more on false negatives
    elif config.loss_function == "iou":
        return IoULoss()
    elif config.loss_function == "bce":
        return nn.BCEWithLogitsLoss()
    elif config.loss_function == "combined":
        losses = {
            "dice": DiceLoss(),
            "focal": FocalLoss(),
            "bce": nn.BCEWithLogitsLoss()
        }
        weights = {"dice": 0.4, "focal": 0.4, "bce": 0.2}
        return CombinedLoss(losses, weights)
    else:
        raise ValueError(f"Unsupported loss function: {config.loss_function}")


class SegmentationMetrics:
    """Collection of segmentation metrics"""
    
    @staticmethod
    def dice_coefficient(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        predictions = (torch.sigmoid(predictions) > threshold).float()
        targets = targets.float()
        
        intersection = (predictions * targets).sum()
        total = predictions.sum() + targets.sum()
        
        if total == 0:
            return 1.0
        
        return (2.0 * intersection / total).item()
    
    @staticmethod
    def iou_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        predictions = (torch.sigmoid(predictions) > threshold).float()
        targets = targets.float()
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        if union == 0:
            return 1.0
        
        return (intersection / union).item()
    
    @staticmethod
    def pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        predictions = (torch.sigmoid(predictions) > threshold).float()
        targets = targets.float()
        
        correct = (predictions == targets).float().sum()
        total = targets.numel()
        
        return (correct / total).item()
    
    @staticmethod
    def sensitivity_recall(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        predictions = (torch.sigmoid(predictions) > threshold).float()
        targets = targets.float()
        
        tp = (predictions * targets).sum()
        fn = ((1 - predictions) * targets).sum()
        
        if tp + fn == 0:
            return 1.0
        
        return (tp / (tp + fn)).item()
    
    @staticmethod
    def specificity(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        predictions = (torch.sigmoid(predictions) > threshold).float()
        targets = targets.float()
        
        tn = ((1 - predictions) * (1 - targets)).sum()
        fp = (predictions * (1 - targets)).sum()
        
        if tn + fp == 0:
            return 1.0
        
        return (tn / (tn + fp)).item() 
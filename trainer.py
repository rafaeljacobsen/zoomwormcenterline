import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import Optional, Dict, Any
import numpy as np

from models import create_model, get_model_info
from losses import get_loss_function, SegmentationMetrics
from dataset import create_data_loaders


class SegmentationLightningModule(pl.LightningModule):
    """PyTorch Lightning module for segmentation training"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model and loss function
        self.model = create_model(config)
        self.loss_fn = get_loss_function(config)
        
        # Initialize metrics
        self.metrics = SegmentationMetrics()
        
        # For tracking best validation metrics
        self.best_val_dice = 0.0
        
        # Log model info
        model_info = get_model_info(self.model)
        print(f"Model: {config.model_name}")
        print(f"Total parameters: {model_info['total_parameters']:,}")
        print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        
        # Create optimizer
        if self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Create scheduler
        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None
        
        if scheduler is None:
            return optimizer
        elif self.config.scheduler == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_dice",
                    "frequency": 1
                }
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1
                }
            }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        images, masks = batch
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate loss
        loss = self.loss_fn(outputs, masks)
        
        # Calculate metrics
        dice = self.metrics.dice_coefficient(outputs, masks)
        iou = self.metrics.iou_score(outputs, masks)
        pixel_acc = self.metrics.pixel_accuracy(outputs, masks)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True)
        self.log('train_pixel_acc', pixel_acc, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, masks = batch
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate loss
        loss = self.loss_fn(outputs, masks)
        
        # Calculate metrics
        dice = self.metrics.dice_coefficient(outputs, masks)
        iou = self.metrics.iou_score(outputs, masks)
        pixel_acc = self.metrics.pixel_accuracy(outputs, masks)
        sensitivity = self.metrics.sensitivity_recall(outputs, masks)
        specificity = self.metrics.specificity(outputs, masks)
        
        # Log metrics
        metrics_dict = {
            'val_loss': loss,
            'val_dice': dice,
            'val_iou': iou,
            'val_pixel_acc': pixel_acc,
            'val_sensitivity': sensitivity,
            'val_specificity': specificity
        }
        
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        
        return metrics_dict
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        images, masks = batch
        
        # Forward pass (with test-time augmentation if enabled)
        if self.config.test_time_augmentation:
            outputs = self.predict_with_tta(images)
        else:
            outputs = self.model(images)
        
        # Calculate loss
        loss = self.loss_fn(outputs, masks)
        
        # Calculate metrics
        dice = self.metrics.dice_coefficient(outputs, masks)
        iou = self.metrics.iou_score(outputs, masks)
        pixel_acc = self.metrics.pixel_accuracy(outputs, masks)
        sensitivity = self.metrics.sensitivity_recall(outputs, masks)
        specificity = self.metrics.specificity(outputs, masks)
        
        # Log metrics
        metrics_dict = {
            'test_loss': loss,
            'test_dice': dice,
            'test_iou': iou,
            'test_pixel_acc': pixel_acc,
            'test_sensitivity': sensitivity,
            'test_specificity': specificity
        }
        
        self.log_dict(metrics_dict, on_step=False, on_epoch=True)
        
        return metrics_dict
    
    def predict_with_tta(self, images):
        """Test-time augmentation for better predictions"""
        batch_size, channels, height, width = images.shape
        
        # Original prediction
        outputs = [self.model(images)]
        
        # Horizontal flip
        if self.config.tta_transforms >= 2:
            flipped_h = torch.flip(images, dims=[3])
            pred_h = self.model(flipped_h)
            pred_h = torch.flip(pred_h, dims=[3])
            outputs.append(pred_h)
        
        # Vertical flip
        if self.config.tta_transforms >= 3:
            flipped_v = torch.flip(images, dims=[2])
            pred_v = self.model(flipped_v)
            pred_v = torch.flip(pred_v, dims=[2])
            outputs.append(pred_v)
        
        # Both flips
        if self.config.tta_transforms >= 4:
            flipped_hv = torch.flip(images, dims=[2, 3])
            pred_hv = self.model(flipped_hv)
            pred_hv = torch.flip(pred_hv, dims=[2, 3])
            outputs.append(pred_hv)
        
        # Average all predictions
        return torch.mean(torch.stack(outputs), dim=0)
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Update best validation dice
        current_dice = self.trainer.callback_metrics.get('val_dice', 0)
        if current_dice > self.best_val_dice:
            self.best_val_dice = current_dice
        
        self.log('best_val_dice', self.best_val_dice, prog_bar=True)


def train_model(config):
    """Main training function"""
    
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create Lightning module
    model = SegmentationLightningModule(config)
    
    # Create callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.save_dir,
        filename='best-{epoch:02d}-{val_dice:.4f}',
        monitor='val_dice',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_dice',
        mode='max',
        patience=20,
        verbose=True,
        min_delta=0.001
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Logger
    logger = None
    if config.use_wandb:
        logger = WandbLogger(
            project=config.project_name,
            name=f"{config.model_name}_{config.encoder_name}",
            config=config.__dict__
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,  # Mixed precision
        log_every_n_steps=config.log_every_n_steps,
        deterministic=True,
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    print("Testing model...")
    trainer.test(model, test_loader, ckpt_path='best')
    
    # Save final model
    final_model_path = os.path.join(config.save_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    if config.use_wandb:
        wandb.finish()
    
    return model, trainer


def load_trained_model(checkpoint_path: str, config):
    """Load a trained model from checkpoint"""
    model = SegmentationLightningModule.load_from_checkpoint(
        checkpoint_path, 
        config=config
    )
    model.eval()
    return model


if __name__ == "__main__":
    from config import Config
    
    # Create config
    config = Config()
    
    # Train model
    model, trainer = train_model(config) 
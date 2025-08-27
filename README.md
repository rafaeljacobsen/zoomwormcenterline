# 🐛 Worm Segmentation with Deep Learning

A modern, production-ready deep learning pipeline for worm segmentation using state-of-the-art neural networks. This project implements multiple segmentation architectures with advanced training strategies for excellent performance on biological image data.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## ✨ Features

### 🎯 **Modern Architectures**
- **U-Net++**: Nested skip connections for precise segmentation
- **DeepLabV3+**: Atrous convolutions with multi-scale features
- **U-Net**: Classic encoder-decoder with skip connections
- **FPN**: Feature Pyramid Network for multi-scale detection
- **PSPNet**: Pyramid Scene Parsing for context understanding
- **MAnet**: Multi-scale Attention Network

### 🔧 **Advanced Training**
- **PyTorch Lightning**: Modern, scalable training framework
- **Mixed Precision**: Faster training with automatic mixed precision
- **Multiple Loss Functions**: Dice, Focal, Tversky, IoU, and combinations
- **Smart Optimizers**: AdamW, Adam, SGD with advanced schedulers
- **Early Stopping**: Automatic training termination with patience
- **Gradient Clipping**: Training stability and convergence

### 📊 **Data Pipeline**
- **Robust Augmentations**: Albumentations with 10+ transformations
- **Automatic Data Splitting**: Train/validation/test with stratification
- **Test-Time Augmentation**: 4x inference improvement with TTA
- **Efficient Loading**: Multi-threaded data loading with caching

### 🎨 **Visualization & Analysis**
- **Dataset Explorer**: Comprehensive data analysis and recommendations
- **Training Monitoring**: Weights & Biases integration
- **Real-time Metrics**: Dice, IoU, sensitivity, specificity
- **Interactive Visualizations**: Side-by-side comparisons and overlays

## 🚀 Quick Start

### One-Click Installation

#### Linux/WSL (Recommended)
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/worm-segmentation.git
cd worm-segmentation

# Run automatic installation
chmod +x install.sh
./install.sh
```

#### Windows
```cmd
# Clone the repository
git clone https://github.com/YOUR_USERNAME/worm-segmentation.git
cd worm-segmentation

# Run installation (as Administrator)
install.bat
```

#### Manual Installation
See [README_INSTALL.md](README_INSTALL.md) for detailed installation instructions.

### Quick Usage

```bash
# Activate environment
conda activate zoomworm

# 1. Explore your dataset
python explore_data.py --visualize

# 2. Train the model (quick test)
python main.py --epochs 10 --batch-size 4 --no-wandb

# 3. Train the model (full training)
python main.py --epochs 100 --batch-size 8

# 4. Run inference on images
python main.py --mode inference \
    --checkpoint checkpoints/best-model.ckpt \
    --input-folder /path/to/images \
    --output-folder results

# 5. Process video (fastest)
python video_inference.py \
    --video /path/to/video.mp4 \
    --checkpoint checkpoints/best-model.ckpt \
    --output video_results \
    --batch-size 32
```

## 📋 Configuration Options

### Model Architectures
- `UnetPlusPlus` - Best overall performance (recommended)
- `DeepLabV3Plus` - Great for complex boundaries  
- `Unet` - Fast and reliable baseline
- `FPN` - Good for multi-scale objects
- `PSPNet` - Strong context understanding
- `MAnet` - Attention-based segmentation

### Backbone Encoders
- `efficientnet-b4` - Best accuracy/speed balance (recommended)
- `efficientnet-b2` - Faster, good for smaller datasets
- `resnet50` - Reliable baseline
- `densenet121` - Strong feature reuse
- And 100+ more from timm library

### Loss Functions
- `dice_focal` - Handles class imbalance (recommended)
- `dice` - Good for balanced datasets
- `focal` - Best for very sparse masks
- `tversky` - Emphasis on false negatives
- `combined` - Multiple loss combination

## 🏗️ Project Structure

```
worm-segmentation/
├── config.py              # Configuration management
├── dataset.py              # Data loading and augmentation
├── models.py               # Neural network architectures
├── losses.py               # Loss functions and metrics
├── trainer.py              # PyTorch Lightning training
├── inference.py            # Model inference and post-processing
├── video_inference.py      # Fast video processing
├── main.py                 # Main training script
├── explore_data.py         # Dataset analysis tool
├── install.sh              # Linux installation script
├── install.bat             # Windows installation script
├── requirements.txt        # Python dependencies
├── README_INSTALL.md       # Detailed installation guide
└── README.md               # This file
```

## 🎯 Training Examples

### Quick Training
```bash
# Fast training for testing
python main.py --epochs 10 --batch-size 4 --no-wandb
```

### High-Performance Training
```bash
# Full training for best results
python main.py \
    --model UnetPlusPlus \
    --encoder efficientnet-b4 \
    --img-size 640 640 \
    --batch-size 6 \
    --loss dice_focal \
    --epochs 150 \
    --lr 1e-4
```

### Multi-GPU Training
```bash
# Automatic multi-GPU with PyTorch Lightning
python main.py --batch-size 16  # Will auto-distribute
```

## 📊 Performance Results

With proper configuration, you should achieve:
- **Dice Score**: 0.90-0.95+ on well-annotated data
- **Training Time**: 2-4 hours on modern GPU
- **Inference Speed**: 10-50 FPS depending on size
- **Memory Usage**: 4-8GB GPU memory

## 🛠️ Advanced Usage

### Custom Data Paths
```bash
python main.py \
    --data-root /custom/path \
    --img-dir my_images \
    --mask-dir my_masks
```

### Experiment with Different Models
```bash
# Try different architectures
python main.py --model DeepLabV3Plus --encoder resnet50
python main.py --model FPN --encoder efficientnet-b2
python main.py --model MAnet --encoder densenet121
```

### Hyperparameter Tuning
```bash
# Learning rate sweep
python main.py --lr 1e-3
python main.py --lr 5e-4
python main.py --lr 1e-4

# Loss function comparison
python main.py --loss dice
python main.py --loss focal
python main.py --loss tversky
```

## 📈 Performance Tips

### Memory Optimization
- Reduce batch size for large images
- Use gradient accumulation for effective large batches
- Enable mixed precision training (automatic)

### Speed Optimization
- Use smaller encoders (efficientnet-b2 vs b4)
- Use video processing for multiple frames
- Increase batch size if GPU memory allows

### Accuracy Optimization
- Use larger images (768x768 or 1024x1024)
- Enable test-time augmentation
- Try ensemble of multiple models
- Increase training epochs

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size
python main.py --batch-size 4

# Use smaller image size
python main.py --img-size 384 384
```

**Poor Performance**
```bash
# Check data with explorer
python explore_data.py --visualize

# Try different loss function
python main.py --loss focal  # for sparse masks
python main.py --loss dice   # for balanced masks
```

**CUDA Issues (WSL)**
```bash
# Set up CUDA environment
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
```

For detailed troubleshooting, see [README_INSTALL.md](README_INSTALL.md).

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) for model implementations
- [PyTorch Lightning](https://lightning.ai/) for training framework
- [Albumentations](https://albumentations.ai/) for data augmentation
- [Weights & Biases](https://wandb.ai/) for experiment tracking

## 📞 Support

If you encounter issues:
1. Check the [installation guide](README_INSTALL.md)
2. Look at [common issues](#troubleshooting)
3. Search existing GitHub issues
4. Create a new issue with detailed information

---

**Ready to train your worm segmentation model? Start with the [installation guide](README_INSTALL.md)! 🚀** 
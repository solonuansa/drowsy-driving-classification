# Drowsy Driving Classification

A deep learning project for detecting driver drowsiness from images using PyTorch and convolutional neural networks (CNNs).

## Overview

This project implements transfer learning to classify driver states as **Drowsy** or **Non Drowsy**. The notebook compares three lightweight pre-trained architectures optimized for real-time performance:

- **MobileNetV3** - Efficient architecture designed for mobile devices
- **EfficientNet-B0** - Balanced accuracy and efficiency
- **SqueezeNet** - Compact model with fewer parameters

The models are fine-tuned on a drowsiness detection dataset and evaluated on accuracy, confusion matrices, and inference latency.

## Dataset

Download the Drowsy Driving dataset from [Kaggle](https://www.kaggle.com/datasets/akshaybhalotia/drowsy-driving-classification).

Update the `main_path` variable in [code.ipynb](code.ipynb) to point to your dataset location:

```python
main_path = "/path/to/Driver Drowsiness Dataset (DDD)/"
```

The dataset is split into:
- Training: ~70%
- Validation: ~20%
- Test: ~10%

## Installation

Install required dependencies:

```powershell
pip install torch torchvision torchsummary numpy pandas matplotlib scikit-learn pillow
```

The notebook automatically uses GPU if available.

## Usage

1. Open [code.ipynb](code.ipynb) in Jupyter or VS Code
2. Update the dataset path
3. Run cells to train and evaluate all three models

The notebook includes:
- Data augmentation and preprocessing
- Model training with early stopping
- Performance evaluation and visualization
- Inference latency measurement

## Results

The project outputs:
- Trained model checkpoints: `best_mobilenet_model.pth`, `best_efficientnet_model.pth`, `best_squeezenet_model.pth`
- Training and validation loss curves
- Confusion matrices for each model
- Latency comparison plots

## License

See [LICENSE](LICENSE) file for details.

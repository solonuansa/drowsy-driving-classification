# drowsy-driving-classification

This repository contains a PyTorch notebook and supporting code for detecting driver drowsiness from images using convolutional neural networks (CNNs).

## Overview

The included notebook `code.ipynb` implements transfer learning experiments and evaluation for binary classification (Drowsy vs Non Drowsy). Instead of a single model, the project compares three pre-trained architectures adapted for this task:

- MobileNetV3 (mobilenet_v3_large)
- EfficientNet-B0 (efficientnet_b0)
- SqueezeNet (squeezenet1_0)

Key steps performed in the notebook:

- Load and split the dataset into train / val / test (≈70% / 20% / 10%).
- Apply image transforms and augmentations (resize to 224×224, color jitter, normalize with ImageNet mean/std).
- Prepare PyTorch `Dataset` and `DataLoader` (default batch_size=64 in the notebook).
- Initialize and fine-tune pretrained models by replacing classifier heads and unfreezing the last layers.
- Train with cross-entropy loss and Adam optimizer, use checkpointing and early stopping.
- Evaluate using accuracy, loss, confusion matrices and plot training curves.
- Measure inference latency for saved models.

## Dataset

The notebook expects the Drowsy Driving dataset. A publicly available dataset is linked on Kaggle:

- https://www.kaggle.com/datasets/akshaybhalotia/drowsy-driving-classification

In the notebook the dataset path is set to:

`/kaggle/input/driver-drowsiness-dataset-ddd/Driver Drowsiness Dataset (DDD)/`

If you run the notebook locally, download and extract the dataset and update the `main_path` variable in `code.ipynb` to point to your local dataset folder.

## Dependencies

The notebook uses the following Python packages (tested with recent PyTorch / torchvision releases):

- torch
- torchvision
- torchsummary
- numpy
- pandas
- matplotlib
- scikit-learn
- pillow

Install via pip, for example:

```powershell
pip install torch torchvision torchsummary numpy pandas matplotlib scikit-learn pillow
```

Note: Install the correct CUDA-enabled torch build for your GPU if needed. The notebook automatically selects GPU when available.

## How to run

1. Open `code.ipynb` in Jupyter / VS Code and update the dataset `main_path` if running locally.
2. Ensure dependencies are installed and the Python kernel has access to PyTorch.
3. Run cells interactively. Training, evaluation, and latency-measurement utilities are implemented in the notebook.

The notebook saves best model checkpoints named like `best_<model_name>_model.pth` (in the notebook they are saved under `/kaggle/working/` by default).

## Outputs

- Trained model checkpoints (`best_mobilenet_model.pth`, `best_efficientnet_model.pth`, `best_squeezenet_model.pth`).
- Training / validation loss plots, confusion matrices and a small latency comparison plot.

## License

See the `LICENSE` file in this repository for license details.
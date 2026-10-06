# CIFAR-10 Image Classification with PyTorch

A deep learning project that trains a ResNet-style convolutional neural network
to classify 32x32 colour images into 10 classes (airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck), then visualizes the results.

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch, torchvision |
| Dataset | CIFAR-10 (60,000 images; auto-downloaded) |
| Evaluation | scikit-learn (confusion matrix, classification report) |
| Visualization | Matplotlib, Seaborn |
| Hardware | CPU, NVIDIA GPU (CUDA) or Apple Silicon (MPS), auto-detected |

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and generate all visualizations
python train.py --epochs 15
```

The dataset downloads automatically into `./data` on first run.

Tip: no GPU? Use Google Colab (Runtime > Change runtime type > GPU), upload the
folder and run `!python train.py --epochs 15`. CPU-only: use `--epochs 5` to test quickly.

## Project Pipeline

1. **Data**: 45k train / 5k validation / 10k test split. Training augmentation uses
   random crop and horizontal flip; images are normalized per channel.
2. **Model**: 3 conv stages (Conv-BN-ReLU-MaxPool + residual block), global average
   pooling, dropout, linear classifier (~1.2M parameters).
3. **Training**: SGD with Nesterov momentum, OneCycle learning-rate schedule, weight
   decay, label smoothing. Best checkpoint is chosen by validation accuracy.
4. **Evaluation**: accuracy, per-class precision/recall/F1 on the held-out test set.

## Output (saved in `results/`)

| File | What it shows |
|---|---|
| `sample_images.png` | Augmented training samples |
| `training_curves.png` | Loss and accuracy per epoch (spot over/underfitting) |
| `confusion_matrix.png` | Which classes get confused (e.g., cat vs dog) |
| `per_class_accuracy.png` | Accuracy for each class |
| `predictions.png` | Random predictions with confidence, green/red for correct/wrong |
| `misclassified.png` | Most confident mistakes |
| `classification_report.txt` | Precision, recall, F1 per class |
| `model.pt` | Trained weights |

Expected test accuracy is roughly 88-92% after 15-20 epochs on a GPU.

## Load the trained model

```python
import torch
from train import CNN
model = CNN(); model.load_state_dict(torch.load("results/model.pt", map_location="cpu")); model.eval()
```

## Ideas to Extend

- Swap in transfer learning (`torchvision.models.resnet18(weights="DEFAULT")`)
- Add Grad-CAM heatmaps to explain predictions
- Try mixup/cutmix augmentation or a learning-rate finder
- Wrap the model in a Streamlit or Gradio app for image upload

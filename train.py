"""
CIFAR-10 Image Classification with a CNN (PyTorch)

Trains a small ResNet-style CNN, evaluates it, and saves visualizations:
  results/sample_images.png        - raw training samples
  results/training_curves.png      - loss and accuracy per epoch
  results/confusion_matrix.png     - normalized confusion matrix
  results/per_class_accuracy.png   - accuracy for each class
  results/predictions.png          - correct (green) vs wrong (red) predictions
  results/misclassified.png        - most confident mistakes
  results/classification_report.txt
  results/model.pt                 - trained weights

Usage:
  python train.py --epochs 15
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


# ----------------------------- Model -----------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        def stage(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), ResBlock(cout))
        self.features = nn.Sequential(stage(3, 64), stage(64, 128), stage(128, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


# ----------------------------- Data ------------------------------
def get_loaders(batch_size, data_dir="data"):
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    full_train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    val_base = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=test_tf)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(len(full_train)), [45000, 5000], generator=g)
    train_set = torch.utils.data.Subset(full_train, train_idx.indices)
    val_set = torch.utils.data.Subset(val_base, val_idx.indices)

    kw = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    return (DataLoader(train_set, shuffle=True, **kw),
            DataLoader(val_set, **kw), DataLoader(test_set, **kw))


def denorm(img):
    img = img.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * np.array(STD) + np.array(MEAN), 0, 1)


# --------------------------- Training ----------------------------
def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, images = [], [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu())
        labels.append(y)
        images.append(x)
    return torch.cat(probs), torch.cat(labels), torch.cat(images)


# -------------------------- Visualizations -----------------------
def plot_samples(loader, path):
    x, y = next(iter(loader))
    fig, axes = plt.subplots(3, 8, figsize=(14, 6))
    for ax, img, lbl in zip(axes.ravel(), x, y):
        ax.imshow(denorm(img)); ax.set_title(CLASSES[lbl], fontsize=9); ax.axis("off")
    fig.suptitle("Sample training images (after augmentation)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_curves(h, path):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, h["train_loss"], label="train"); ax[0].plot(ep, h["val_loss"], label="validation")
    ax[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy"); ax[0].legend()
    ax[1].plot(ep, h["train_acc"], label="train"); ax[1].plot(ep, h["val_acc"], label="validation")
    ax[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES,
                yticklabels=CLASSES, ax=ax)
    ax.set(title="Normalized confusion matrix (test set)", xlabel="Predicted", ylabel="True")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_per_class(y_true, y_pred, path):
    acc = [(y_pred[y_true == c] == c).mean() for c in range(10)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(CLASSES, acc, color=sns.color_palette("viridis", 10))
    ax.set(ylim=(0, 1), ylabel="Accuracy", title="Per-class test accuracy")
    for i, a in enumerate(acc):
        ax.text(i, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_predictions(images, y_true, y_pred, probs, path, n=16):
    idx = np.random.RandomState(0).choice(len(y_true), n, replace=False)
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    for ax, i in zip(axes.ravel(), idx):
        ok = y_true[i] == y_pred[i]
        ax.imshow(denorm(images[i])); ax.axis("off")
        ax.set_title(f"{CLASSES[y_pred[i]]}\n{probs[i].max():.0%}", fontsize=9,
                     color="green" if ok else "red")
    fig.suptitle("Random test predictions (green = correct, red = wrong)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_misclassified(images, y_true, y_pred, probs, path, n=16):
    wrong = np.where(y_true != y_pred)[0]
    worst = wrong[np.argsort(-probs[wrong].max(1).values.numpy())][:n]
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    for ax, i in zip(axes.ravel(), worst):
        ax.imshow(denorm(images[i])); ax.axis("off")
        ax.set_title(f"true: {CLASSES[y_true[i]]}\npred: {CLASSES[y_pred[i]]}", fontsize=8, color="red")
    fig.suptitle("Most confident mistakes")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------ Main -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_loaders(args.batch_size)
    plot_samples(train_loader, f"{args.out}/sample_images.png")

    model = CNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 5, epochs=args.epochs, steps_per_epoch=len(train_loader))

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, device, criterion, optimizer, scheduler)
        vl, va = run_epoch(model, val_loader, device, criterion)
        for k, v in zip(hist, [tl, ta, vl, va]):
            hist[k].append(v)
        if va > best_val:
            best_val = va
            torch.save(model.state_dict(), f"{args.out}/model.pt")
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.3f} acc {ta:.3f} | "
              f"val loss {vl:.3f} acc {va:.3f} | {time.time() - t0:.0f}s")

    # Evaluate best checkpoint on the test set
    model.load_state_dict(torch.load(f"{args.out}/model.pt", map_location=device))
    probs, labels, images = predict(model, test_loader, device)
    y_pred, y_true = probs.argmax(1).numpy(), labels.numpy()
    test_acc = (y_pred == y_true).mean()
    print(f"\nTest accuracy: {test_acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
    print(report)
    with open(f"{args.out}/classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n\n{report}")

    plot_curves(hist, f"{args.out}/training_curves.png")
    plot_confusion(y_true, y_pred, f"{args.out}/confusion_matrix.png")
    plot_per_class(y_true, y_pred, f"{args.out}/per_class_accuracy.png")
    plot_predictions(images, y_true, y_pred, probs, f"{args.out}/predictions.png")
    plot_misclassified(images, y_true, y_pred, probs, f"{args.out}/misclassified.png")
    print(f"Saved model and plots to ./{args.out}/")


if __name__ == "__main__":
    main()

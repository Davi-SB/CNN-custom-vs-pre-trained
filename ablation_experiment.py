"""Estudo de ablação enxuto da LightCNN.

Compara a CNN da APS anterior com e sem Batch Normalization nos datasets
MNIST e CIFAR-10, usando GPU quando disponível.
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


OUTPUT_DIR = "results_ablation"
BATCH_SIZE = 128
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()


class LightCNNAblation(nn.Module):
    """LightCNN configurável para ligar/desligar Batch Normalization."""

    def __init__(self, in_channels=1, num_classes=10, img_size=28, use_batchnorm=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        self.features = nn.Sequential(
            self._conv_block(in_channels, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
        )

        feat_size = img_size // 8
        flat_features = 128 * feat_size * feat_size
        self.classifier = nn.Sequential(
            nn.Linear(flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _conv_block(self, in_channels, out_channels):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_model(model, train_loader, test_loader, epochs, model_name, scheduler=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["test_loss"].append(round(test_loss, 4))
        history["test_acc"].append(round(test_acc, 4))

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}",
            flush=True,
        )

    history["elapsed_time"] = round(time.time() - start, 1)
    print(f"[{model_name}] concluído em {history['elapsed_time']}s\n", flush=True)
    return history


def get_loaders(dataset_name):
    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        return (
            DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            ),
            DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            ),
            1,
            28,
        )

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
        DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
        3,
        32,
    )


def plot_curves(results, dataset_name):
    pretty_name = "MNIST" if dataset_name == "mnist" else "CIFAR-10"
    with_bn = results[dataset_name]["with_batchnorm"]["history"]
    without_bn = results[dataset_name]["without_batchnorm"]["history"]

    epochs = range(1, len(with_bn["test_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, with_bn["test_loss"], marker="o", label="Com BatchNorm")
    axes[0].plot(epochs, without_bn["test_loss"], marker="s", label="Sem BatchNorm")
    axes[0].set_title(f"Loss de Teste - {pretty_name}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [a * 100 for a in with_bn["test_acc"]], marker="o", label="Com BatchNorm")
    axes[1].plot(epochs, [a * 100 for a in without_bn["test_acc"]], marker="s", label="Sem BatchNorm")
    axes[1].set_title(f"Acurácia de Teste - {pretty_name}")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_bar(results):
    labels = ["MNIST\nCom BN", "MNIST\nSem BN", "CIFAR-10\nCom BN", "CIFAR-10\nSem BN"]
    values = [
        results["mnist"]["with_batchnorm"]["history"]["test_acc"][-1] * 100,
        results["mnist"]["without_batchnorm"]["history"]["test_acc"][-1] * 100,
        results["cifar10"]["with_batchnorm"]["history"]["test_acc"][-1] * 100,
        results["cifar10"]["without_batchnorm"]["history"]["test_acc"][-1] * 100,
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#1976D2", "#90CAF9", "#F57C00", "#FFCC80"])
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Acurácia Final com e sem Batch Normalization")
    ax.set_ylabel("Acurácia de Teste (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()


def run_experiment(dataset_name, epochs):
    train_loader, test_loader, in_channels, img_size = get_loaders(dataset_name)
    dataset_results = {}

    for use_batchnorm, key, label in [
        (True, "with_batchnorm", "Com BatchNorm"),
        (False, "without_batchnorm", "Sem BatchNorm"),
    ]:
        model = LightCNNAblation(
            in_channels=in_channels,
            num_classes=10,
            img_size=img_size,
            use_batchnorm=use_batchnorm,
        ).to(DEVICE)
        param_count = sum(p.numel() for p in model.parameters())
        model_name = f"{dataset_name.upper()} - {label}"
        print(f"\n>>> {model_name} | Parâmetros: {param_count:,}", flush=True)

        history = train_model(model, train_loader, test_loader, epochs, model_name)
        dataset_results[key] = {
            "params": param_count,
            "history": history,
        }

    return dataset_results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Dispositivo: {DEVICE}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch: {torch.__version__}\n", flush=True)

    results = {
        "mnist": run_experiment("mnist", epochs=5),
        "cifar10": run_experiment("cifar10", epochs=10),
    }

    plot_curves(results, "mnist")
    plot_curves(results, "cifar10")
    plot_bar(results)

    with open(os.path.join(OUTPUT_DIR, "results_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nResumo final")
    for dataset_name, dataset_results in results.items():
        pretty_name = "MNIST" if dataset_name == "mnist" else "CIFAR-10"
        for key, label in [
            ("with_batchnorm", "Com BatchNorm"),
            ("without_batchnorm", "Sem BatchNorm"),
        ]:
            history = dataset_results[key]["history"]
            print(
                f"{pretty_name} - {label}: "
                f"Acc={history['test_acc'][-1] * 100:.2f}% | "
                f"Loss={history['test_loss'][-1]:.4f} | "
                f"Tempo={history['elapsed_time']}s"
            )


if __name__ == "__main__":
    main()

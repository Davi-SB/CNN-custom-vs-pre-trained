"""
Script para executar todos os experimentos CNN e salvar resultados.
Otimizado para execução em CPU.
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from models import LightCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 0
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
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
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def train_model(model, train_loader, test_loader, criterion, optimizer,
                epochs, scheduler=None, name="Model"):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start = time.time()
    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
        vl, va = evaluate(model, test_loader, criterion)
        if scheduler:
            scheduler.step()
        history["train_loss"].append(round(tl, 4))
        history["train_acc"].append(round(ta, 4))
        history["val_loss"].append(round(vl, 4))
        history["val_acc"].append(round(va, 4))
        print(f"  [{name}] Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f}",
              flush=True)
    elapsed = time.time() - start
    history["elapsed_time"] = round(elapsed, 1)
    print(f"  [{name}] Concluido em {elapsed:.1f}s\n", flush=True)
    return history


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        outputs = model(images.to(DEVICE))
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def save_training_curves(hist, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(hist["train_loss"]) + 1)
    ax1.plot(epochs_range, hist["train_loss"], "o-", markersize=3, label="Treino")
    ax1.plot(epochs_range, hist["val_loss"], "s-", markersize=3, label="Validacao")
    ax1.set_xlabel("Epoca"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss - {title}"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs_range, [a*100 for a in hist["train_acc"]], "o-", markersize=3, label="Treino")
    ax2.plot(epochs_range, [a*100 for a in hist["val_acc"]], "s-", markersize=3, label="Validacao")
    ax2.set_xlabel("Epoca"); ax2.set_ylabel("Acuracia (%)")
    ax2.set_title(f"Acuracia - {title}"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title, fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def save_comparison_bar(results_dict, title, filename):
    names = list(results_dict.keys())
    accs = [results_dict[n]["val_acc"][-1] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accs, color=["#2196F3", "#FF9800"])
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Acuracia (%)"); ax.set_title(title)
    ax.set_ylim(0, 105); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def save_overlay_curves(results_dict, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"LightCNN": "#2196F3", "ResNet18": "#FF9800"}
    for name, hist in results_dict.items():
        c = colors.get(name, "gray")
        epochs = range(1, len(hist["val_loss"]) + 1)
        ax1.plot(epochs, hist["val_loss"], "o-", markersize=3, label=name, color=c)
        ax2.plot(epochs, [a*100 for a in hist["val_acc"]], "o-", markersize=3, label=name, color=c)
    ax1.set_xlabel("Epoca"); ax1.set_ylabel("Loss (Validacao)")
    ax1.set_title(f"Loss de Validacao - {title}"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Epoca"); ax2.set_ylabel("Acuracia (%) (Validacao)")
    ax2.set_title(f"Acuracia de Validacao - {title}"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def save_activations(model, image, title_prefix, filename_prefix):
    model.eval()
    x = image.to(DEVICE)
    conv_outputs, conv_names = [], []
    idx = 0
    for layer in model.features:
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            conv_outputs.append(x.cpu().squeeze(0))
            conv_names.append(f"Conv2d #{idx + 1}")
            idx += 1
    for feat_map, name in zip(conv_outputs, conv_names):
        num_filters = min(feat_map.shape[0], 16)
        cols, rows = 8, max(1, (num_filters + 7) // 8)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        fig.suptitle(f"{title_prefix}{name} - Ativacoes dos Filtros", fontsize=12)
        if rows == 1 and cols > 1:
            axes = [axes]
        elif rows == 1 and cols == 1:
            axes = [[axes]]
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            if i < num_filters:
                ax.imshow(feat_map[i].numpy(), cmap="viridis")
            ax.axis("off")
        plt.tight_layout()
        fn = f"{filename_prefix}_{name.replace(' ', '').replace('#', '')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fn), dpi=150, bbox_inches="tight")
        plt.close()


def build_resnet18(num_classes=10):
    """ResNet18 com camadas congeladas exceto layer4 + fc."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# ── MNIST ────────────────────────────────────────────────────────────────────
def run_mnist():
    print("=" * 60)
    print("MNIST")
    print("=" * 60, flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    classes = [str(i) for i in range(10)]

    criterion = nn.CrossEntropyLoss()

    # LightCNN — 10 epochs
    print("\n>>> Treinando LightCNN no MNIST...", flush=True)
    model_cnn = LightCNN(in_channels=1, num_classes=10, img_size=28).to(DEVICE)
    opt_cnn = optim.Adam(model_cnn.parameters(), lr=1e-3)
    n_params_cnn = sum(p.numel() for p in model_cnn.parameters())
    print(f"  Parametros: {n_params_cnn:,}", flush=True)
    hist_cnn = train_model(model_cnn, train_loader, test_loader, criterion, opt_cnn,
                           epochs=10, name="LightCNN")

    # ResNet18 — layer4+fc treinaveis, 10 epochs
    print(">>> Fine-tuning ResNet18 no MNIST...", flush=True)
    transform_rn = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,)*3, (0.3081,)*3),
    ])
    train_ds_rn = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_rn)
    test_ds_rn = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_rn)
    train_loader_rn = DataLoader(train_ds_rn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader_rn = DataLoader(test_ds_rn, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model_rn = build_resnet18(10)
    trainable_rn = sum(p.numel() for p in model_rn.parameters() if p.requires_grad)
    total_rn = sum(p.numel() for p in model_rn.parameters())
    print(f"  Total: {total_rn:,} | Trein.: {trainable_rn:,}", flush=True)
    opt_rn = optim.Adam(filter(lambda p: p.requires_grad, model_rn.parameters()), lr=1e-3)
    hist_rn = train_model(model_rn, train_loader_rn, test_loader_rn, criterion, opt_rn,
                          epochs=10, name="ResNet18")

    # Graficos
    results = {"LightCNN": hist_cnn, "ResNet18": hist_rn}
    save_training_curves(hist_cnn, "LightCNN - MNIST", "mnist_lightcnn_curves.png")
    save_training_curves(hist_rn, "ResNet18 - MNIST", "mnist_resnet18_curves.png")
    save_comparison_bar(results, "Comparacao de Acuracia - MNIST", "mnist_comparison_bar.png")
    save_overlay_curves(results, "MNIST", "mnist_overlay_curves.png")

    # Matrizes de confusao
    yt_c, yp_c = get_predictions(model_cnn, test_loader)
    save_confusion_matrix(yt_c, yp_c, classes, "Matriz de Confusao - LightCNN (MNIST)", "mnist_cm_lightcnn.png")
    yt_r, yp_r = get_predictions(model_rn, test_loader_rn)
    save_confusion_matrix(yt_r, yp_r, classes, "Matriz de Confusao - ResNet18 (MNIST)", "mnist_cm_resnet18.png")

    # Ativacoes (DESAFIO)
    sample, _ = test_ds[0]
    save_activations(model_cnn, sample.unsqueeze(0), "LightCNN MNIST - ", "mnist_act_lightcnn")

    report_cnn = classification_report(yt_c, yp_c, target_names=classes, output_dict=True)
    report_rn = classification_report(yt_r, yp_r, target_names=classes, output_dict=True)

    print(">>> MNIST concluido!", flush=True)
    return {
        "LightCNN": {"params": n_params_cnn, "history": hist_cnn, "report": report_cnn},
        "ResNet18": {"params_total": total_rn, "params_trainable": trainable_rn,
                     "history": hist_rn, "report": report_rn},
    }


# ── CIFAR-10 ─────────────────────────────────────────────────────────────────
def run_cifar10():
    print("=" * 60)
    print("CIFAR-10")
    print("=" * 60, flush=True)

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    criterion = nn.CrossEntropyLoss()

    # LightCNN — 30 epochs
    print("\n>>> Treinando LightCNN no CIFAR-10...", flush=True)
    model_cnn = LightCNN(in_channels=3, num_classes=10, img_size=32).to(DEVICE)
    opt_cnn = optim.Adam(model_cnn.parameters(), lr=1e-3)
    sched_cnn = optim.lr_scheduler.StepLR(opt_cnn, step_size=15, gamma=0.1)
    n_params_cnn = sum(p.numel() for p in model_cnn.parameters())
    print(f"  Parametros: {n_params_cnn:,}", flush=True)
    hist_cnn = train_model(model_cnn, train_loader, test_loader, criterion, opt_cnn,
                           epochs=30, scheduler=sched_cnn, name="LightCNN")

    # ResNet18 — layer4+fc, 15 epochs
    print(">>> Fine-tuning ResNet18 no CIFAR-10...", flush=True)
    model_rn = build_resnet18(10)
    trainable_rn = sum(p.numel() for p in model_rn.parameters() if p.requires_grad)
    total_rn = sum(p.numel() for p in model_rn.parameters())
    print(f"  Total: {total_rn:,} | Trein.: {trainable_rn:,}", flush=True)
    opt_rn = optim.Adam(filter(lambda p: p.requires_grad, model_rn.parameters()), lr=1e-3)
    sched_rn = optim.lr_scheduler.StepLR(opt_rn, step_size=8, gamma=0.1)
    hist_rn = train_model(model_rn, train_loader, test_loader, criterion, opt_rn,
                          epochs=15, scheduler=sched_rn, name="ResNet18")

    # Graficos
    results = {"LightCNN": hist_cnn, "ResNet18": hist_rn}
    save_training_curves(hist_cnn, "LightCNN - CIFAR-10", "cifar10_lightcnn_curves.png")
    save_training_curves(hist_rn, "ResNet18 - CIFAR-10", "cifar10_resnet18_curves.png")
    save_comparison_bar(results, "Comparacao de Acuracia - CIFAR-10", "cifar10_comparison_bar.png")
    save_overlay_curves(results, "CIFAR-10", "cifar10_overlay_curves.png")

    # Matrizes de confusao
    yt_c, yp_c = get_predictions(model_cnn, test_loader)
    save_confusion_matrix(yt_c, yp_c, classes, "Matriz de Confusao - LightCNN (CIFAR-10)", "cifar10_cm_lightcnn.png")
    yt_r, yp_r = get_predictions(model_rn, test_loader)
    save_confusion_matrix(yt_r, yp_r, classes, "Matriz de Confusao - ResNet18 (CIFAR-10)", "cifar10_cm_resnet18.png")

    # Ativacoes (DESAFIO)
    sample, _ = test_ds[0]
    save_activations(model_cnn, sample.unsqueeze(0), "LightCNN CIFAR-10 - ", "cifar10_act_lightcnn")

    report_cnn = classification_report(yt_c, yp_c, target_names=classes, output_dict=True)
    report_rn = classification_report(yt_r, yp_r, target_names=classes, output_dict=True)

    print(">>> CIFAR-10 concluido!", flush=True)
    return {
        "LightCNN": {"params": n_params_cnn, "history": hist_cnn, "report": report_cnn},
        "ResNet18": {"params_total": total_rn, "params_trainable": trainable_rn,
                     "history": hist_rn, "report": report_rn},
    }


if __name__ == "__main__":
    print(f"Dispositivo: {DEVICE}")
    print(f"PyTorch: {torch.__version__}\n", flush=True)

    mnist_results = run_mnist()
    cifar_results = run_cifar10()

    all_results = {"mnist": mnist_results, "cifar10": cifar_results}
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    for ds_name, ds_res in all_results.items():
        print(f"\n--- {ds_name.upper()} ---")
        for model_name, model_res in ds_res.items():
            h = model_res["history"]
            print(f"  {model_name}: Acc Teste = {h['val_acc'][-1]*100:.2f}% | Tempo = {h['elapsed_time']}s")

    print(f"\nResultados salvos em '{OUTPUT_DIR}/'")
    print("Concluido!")

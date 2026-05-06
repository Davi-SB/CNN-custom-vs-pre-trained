import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, test_loader, criterion, optimizer,
                device, epochs, scheduler=None, model_name="Model"):
    """Loop completo de treino com log por época. Retorna histórico."""
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

    elapsed = time.time() - start
    print(f"[{model_name}] Treinamento concluído em {elapsed:.1f}s")
    history["elapsed_time"] = elapsed
    return history


def plot_training_curves(history, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Treino")
    ax1.plot(history["val_loss"], label="Validação")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss - {title}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Treino")
    ax2.plot(history["val_acc"], label="Validação")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia")
    ax2.set_title(f"Acurácia - {title}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_comparison_bar(results, metric="val_acc", ylabel="Acurácia (%)",
                        title="Comparação de Modelos"):
    """Gráfico de barras comparando modelos. results é dict {nome: history}."""
    names = list(results.keys())
    values = [results[n][metric][-1] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values, color=["#2196F3", "#FF9800"])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, classes, title="Matriz de Confusão"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def visualize_activations(model, image, device, layer_indices=None,
                          title_prefix=""):
    """
    Plota os feature maps de saída das camadas convolucionais.
    - model: modelo com atributo 'features' (Sequential)
    - image: tensor (1, C, H, W) já normalizado
    - layer_indices: quais camadas Conv2d visualizar (None = todas)
    """
    model.eval()
    x = image.to(device)

    conv_outputs = []
    conv_names = []
    idx = 0
    for layer in model.features:
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            if layer_indices is None or idx in layer_indices:
                conv_outputs.append(x.cpu().squeeze(0))
                conv_names.append(f"Conv2d #{idx + 1}")
            idx += 1

    for feat_map, name in zip(conv_outputs, conv_names):
        num_filters = min(feat_map.shape[0], 16)
        cols = 8
        rows = (num_filters + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        fig.suptitle(f"{title_prefix}{name} — Ativações dos Filtros",
                     fontsize=12)

        if rows == 1:
            axes = [axes]
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r][c] if rows > 1 else axes[0][c] if cols > 1 else axes[0]
            if i < num_filters:
                ax.imshow(feat_map[i].numpy(), cmap="viridis")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def print_summary_table(results):
    """Imprime tabela comparativa dos resultados."""
    print(f"{'Modelo':<20} {'Acc Treino':>12} {'Acc Teste':>12} {'Tempo (s)':>12}")
    print("-" * 58)
    for name, hist in results.items():
        train_acc = hist["train_acc"][-1] * 100
        val_acc = hist["val_acc"][-1] * 100
        elapsed = hist.get("elapsed_time", 0)
        print(f"{name:<20} {train_acc:>11.2f}% {val_acc:>11.2f}% {elapsed:>11.1f}")

"""Plotting helpers; imported only by report-producing workflows."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, name, path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric, label in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy (%)")):
        for part, title in (("train", "Training (online)"), ("val", "Validation")):
            ax.plot(epochs, [row[f"{part}_{metric}"] for row in history], label=title)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.set_xticks(epochs)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(name)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["val_loss"] for row in history], label=name)
        axes[1].plot(epochs, [row["val_accuracy"] for row in history], label=name)
    for ax, label in zip(axes, ("Validation loss", "Validation accuracy (%)")):
        ax.set(xlabel="Epoch", ylabel=label)
        ax.set_xticks(epochs)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_filters(before, after, path, before_label="Initial", after_label="Trained"):
    """Use one zero-centered scale for all before/after first-layer filters."""
    count = len(before)
    if before.shape != after.shape or before.ndim != 4 or before.shape[1] != 1:
        raise ValueError("Expected matching grayscale first-layer filter arrays.")
    columns = min(count, 8)
    stage_rows = (count + columns - 1) // columns
    limit = max(float(np.abs(before).max()), float(np.abs(after).max()), 1e-12)
    fig, axes = plt.subplots(2 * stage_rows, columns, figsize=(columns * 1.75, stage_rows * 4),
                             squeeze=False, layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for offset, weights, label in ((0, before, before_label), (stage_rows * columns, after, after_label)):
        for index, weight in enumerate(weights):
            ax = axes.ravel()[offset + index]
            image = ax.imshow(weight[0], cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_title(f"{label}: {index}", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Weight; shared scale")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return limit


def plot_confusion(matrix, classes, name, normalized, path):
    """Satir: gercek sinif; sutun: secilen sinif. Kosegen dogru cevaplar."""
    support = matrix.sum(axis=1, keepdims=True)
    values = np.divide(
        matrix, support, out=np.zeros_like(matrix, dtype=float), where=support > 0
    ) if normalized else matrix
    maximum = 1.0 if normalized else max(int(support.max()), 1)
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
    for row in range(len(classes)):
        for col in range(len(classes)):
            label = f"{values[row, col]:.2f}" if normalized else str(matrix[row, col])
            ax.text(col, row, label, ha="center", va="center", fontsize=7,
                    color="white" if values[row, col] > maximum / 2 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted label", ylabel="True label",
           title=f"{name}: {'fraction of each true class' if normalized else 'sample counts'}")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_gallery(raw_dataset, source_indices, targets, predictions, probabilities,
                 positions, title, path):
    """Gorseldeki ID, predictions.csv icindeki kaynak veri indeksiyle aynidir."""
    fig, axes = plt.subplots(2, 5, figsize=(13, 6), layout="constrained")
    classes = raw_dataset.classes
    for ax in axes.ravel():
        ax.axis("off")
    entries = []
    for position, ax in zip(positions, axes.ravel()):
        position = int(position)
        index = int(source_indices[position])
        true_id, predicted_id = int(targets[position]), int(predictions[position])
        score = float(probabilities[position, predicted_id])
        image = raw_dataset.data[index].numpy()
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(
            f"ID {index} | True: {classes[true_id]}\n"
            f"Pred: {classes[predicted_id]} | score: {score:.3f}", fontsize=8,
        )
        entries.append({"dataset_index": index, "true_label": true_id,
                        "predicted_label": predicted_id, "softmax_score": score})
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return entries


def plot_channels(values, name, columns, path):
    rows = (len(values) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2), layout="constrained")
    maximum = max(float(values.max()), 1e-12)
    for ax in axes.ravel():
        ax.axis("off")
    for channel, ax in enumerate(axes.ravel()[:len(values)]):
        image = ax.imshow(values[channel], cmap="viridis", vmin=0, vmax=maximum,
                          interpolation="nearest")
        ax.set_title(f"Channel {channel}", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Activation; shared scale within this figure")
    fig.suptitle(name)
    fig.savefig(path, dpi=150)
    plt.close(fig)

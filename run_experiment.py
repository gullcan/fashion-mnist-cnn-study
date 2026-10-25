"""Fashion-MNIST: iki modeli egit, en iyi agirliklari ve sonuclari kaydet."""

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Grafikleri pencere acmadan dosyaya uret.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from models import build_baseline, build_cnn
from training import evaluate, train_one_epoch


def make_loaders(project_dir, seed, batch_size):
    """Kaydedilmis bolmeyi kullan; her deney icin yeni karistirici kur."""
    with np.load(project_dir / "results" / "split_indices.npz", allow_pickle=False) as split:
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]

    if len(train_indices) != 54000 or len(val_indices) != 6000:
        raise ValueError("Kayitli bolme 54000 egitim / 6000 dogrulama olmali.")
    combined = np.concatenate((train_indices, val_indices))
    if not np.array_equal(np.sort(combined), np.arange(60000)):
        raise ValueError("Her kaynak ornek bolmede tam bir kez bulunmali.")

    dataset = FashionMNIST(
        root=project_dir / "data", train=True, download=False, transform=ToTensor()
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
    )
    return train_loader, val_loader, dataset.classes


def synchronize(device):
    """GPU islemleri bitmeden sure olcumunu sonlandirma."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def plot_filters(before, after, best_epoch, path):
    """Ilk ve secilen modelin filtrelerini ayni renk olceginde goster."""
    limit = max(float(np.abs(before).max()), float(np.abs(after).max()))
    fig, axes = plt.subplots(2, 6, figsize=(13, 5), layout="constrained")
    for row, filters in enumerate((before, after)):
        for index, ax in enumerate(axes[row]):
            picture = ax.imshow(
                filters[index, 0], cmap="coolwarm", vmin=-limit, vmax=limit,
                interpolation="nearest",
            )
            stage = "Initial" if row == 0 else f"Selected epoch {best_epoch}"
            ax.set_title(f"{stage}\nFilter {index}", fontsize=9)
            ax.axis("off")
    fig.colorbar(picture, ax=axes.ravel().tolist(), shrink=0.8, label="Weight")
    fig.suptitle("CNN first-layer weights: before and after training")
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


def run_experiment(project_dir, output_dir, epochs=10, batch_size=64, lr=0.001):
    if epochs < 1:
        raise ValueError("Epoch sayisi en az 1 olmali.")
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = project_dir / "results" / "split_indices.npz"
    with np.load(split_path, allow_pickle=False) as split:
        seed = int(split["seed"].item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = {
        "dataset": "Fashion-MNIST", "train_samples": 54000, "val_samples": 6000,
        "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "optimizer": "Adam", "loss": "CrossEntropyLoss", "num_workers": 0,
        "selection": "minimum validation loss; earlier epoch wins a tie",
        "epoch_seconds_includes": "training and validation; excludes saving and plotting",
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cudnn_deterministic": True, "cudnn_benchmark": False,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    histories, summaries = {}, []
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "epoch_seconds"]
    print(f"Cihaz: {device} | Her model: {epochs} epoch", flush=True)
    print("Satirlardaki loss ve acc sirasi: egitim / dogrulama", flush=True)

    for name, builder in (("linear_baseline", build_baseline), ("cnn", build_cnn)):
        # Baslangic bir kez kurulur; epoch dongusu modeli ve Adam'i sifirlamaz.
        torch.manual_seed(seed)
        model = builder().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        train_loader, val_loader, classes = make_loaders(project_dir, seed, batch_size)
        model_dir = output_dir / name
        model_dir.mkdir(exist_ok=True)
        best_path = model_dir / "best.pt"
        history, best_val_loss = [], float("inf")

        before = None
        if name == "cnn":
            before = model[0].weight.detach().cpu().clone().numpy()
            np.save(model_dir / "filters_before.npy", before)

        with (model_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                synchronize(device)
                started = time.perf_counter()
                train_loss, train_accuracy = train_one_epoch(
                    model, train_loader, loss_fn, optimizer, device
                )
                val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
                synchronize(device)
                elapsed = time.perf_counter() - started
                if not np.isfinite(train_loss + val_loss):
                    raise RuntimeError(f"{name}: sonlu olmayan kayip olustu.")

                row = dict(zip(fields, (
                    epoch, train_loss, train_accuracy, val_loss, val_accuracy, elapsed
                )))
                history.append(row)
                writer.writerow(row)
                file.flush()

                # Yuksek accuracy yerine sabit protokoldeki en dusuk kaybi seciyoruz.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model_name": name, "model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy,
                        "seed": seed, "classes": classes, "config": config,
                    }, best_path)

                print(
                    f"{name} {epoch:02d}/{epochs} | "
                    f"loss {train_loss:.4f}/{val_loss:.4f} | "
                    f"acc {train_accuracy:.2f}/{val_accuracy:.2f}% | {elapsed:.1f}s",
                    flush=True,
                )

        # Rapor ve filtreler son turdan degil, secilen modelden gelir.
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        selected_loss, selected_accuracy = evaluate(model, val_loader, loss_fn, device)
        histories[name] = history
        plot_history(history, name, model_dir / "learning_curves.png")
        if before is not None:
            after = model[0].weight.detach().cpu().clone().numpy()
            np.save(model_dir / "filters_selected.npy", after)
            plot_filters(before, after, checkpoint["epoch"], model_dir / "filters_comparison.png")

        summaries.append({
            "model_name": name, "best_epoch": checkpoint["epoch"],
            "val_loss": selected_loss, "val_accuracy": selected_accuracy,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_and_validation_seconds": sum(row["epoch_seconds"] for row in history),
            "checkpoint": best_path.relative_to(output_dir).as_posix(),
        })

    plot_comparison(histories, output_dir / "validation_comparison.png")
    report = {"config": config, "results": summaries}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nSECILEN MODELLERIN DOGRULAMA OZETI", flush=True)
    for result in summaries:
        print(
            f"{result['model_name']}: epoch={result['best_epoch']} | "
            f"loss={result['val_loss']:.4f} | acc={result['val_accuracy']:.2f}%",
            flush=True,
        )
    print(f"Sonuclar: {output_dir}", flush=True)
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    run_experiment(root, root / "results" / "experiment")

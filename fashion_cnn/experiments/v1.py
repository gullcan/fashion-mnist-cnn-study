"""Linear baseline and original CNN under the recorded V1 protocol."""

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torchvision
from torch import nn

from ..data import make_loaders
from ..engine import evaluate, train_one_epoch, synchronize
from ..models import build_baseline, build_cnn
from ..plotting import plot_history, plot_comparison, plot_filters


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
            plot_filters(before, after, model_dir / "filters_comparison.png",
                         after_label=f"Selected epoch {checkpoint['epoch']}")

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

"""The bounded five-candidate V2 study; selection uses validation only."""

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from ..analysis import analyze_split
from ..artifacts import save_json
from ..data import make_loaders
from ..engine import evaluate, train_one_epoch, synchronize
from ..models import build_cnn
from ..plotting import plot_comparison, plot_history


def better(accuracy, loss, previous_accuracy, previous_loss):
    """V2 hedefi accuracy; esitlikte dusuk loss, tam esitlikte onceki kalir."""
    return accuracy > previous_accuracy or (
        accuracy == previous_accuracy and loss < previous_loss
    )


def train_candidate(project_dir, run_dir, name, parent, config, train_indices,
                    val_indices, seed, epochs, device):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = build_cnn(channels=config["channels"], pooling=config["pooling"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.3
    ) if config["scheduler"] else None
    loss_fn = nn.CrossEntropyLoss()
    train_loader, val_loader, classes = make_loaders(
        project_dir, seed, augmentation=config["augmentation"],
        train_indices=train_indices, val_indices=val_indices
    )
    run_config = {
        "name": name, "parent_run": parent, "candidate_config": config,
        "seed": seed, "epochs": epochs, "batch_size": 64, "initial_lr": 0.001,
        "scheduler": "StepLR(step_size=10,gamma=0.3)" if scheduler else None,
        "augmentation": "pad2/random crop28/horizontal flip0.5" if config["augmentation"] else None,
        "selection": "max validation accuracy; tie: lower loss; exact tie: earlier",
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    save_json(run_dir / "config.json", run_config)
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
              "epoch_seconds", "learning_rate"]
    history, best_accuracy, best_loss = [], -1.0, float("inf")
    best_path = run_dir / "best.pt"

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            # Kaydedilen LR bu turda kullanilandir; scheduler bir sonraki turu ayarlar.
            used_lr = optimizer.param_groups[0]["lr"]
            synchronize(device)
            started = time.perf_counter()
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device
            )
            val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
            synchronize(device)
            elapsed = time.perf_counter() - started
            if not np.isfinite(train_loss + val_loss):
                raise RuntimeError(f"{name}: sonlu olmayan kayip.")
            row = dict(zip(fields, (epoch, train_loss, train_accuracy, val_loss,
                                    val_accuracy, elapsed, used_lr)))
            history.append(row)
            writer.writerow(row)
            file.flush()
            if better(val_accuracy, val_loss, best_accuracy, best_loss):
                best_accuracy, best_loss = val_accuracy, val_loss
                torch.save({
                    "model_name": "cnn_v2", "run_name": name,
                    "model_state_dict": model.state_dict(), "candidate_config": config,
                    "epoch": epoch, "seed": seed, "classes": classes,
                    "val_loss": val_loss, "val_accuracy": val_accuracy,
                    "run_config": run_config,
                }, best_path)
            print(f"{name} {epoch:02d}/{epochs} | lr={used_lr:.6f} | "
                  f"loss={train_loss:.4f}/{val_loss:.4f} | "
                  f"acc={train_accuracy:.2f}/{val_accuracy:.2f}% | {elapsed:.1f}s", flush=True)
            if scheduler is not None:
                scheduler.step()

    saved = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    checked_loss, checked_accuracy = evaluate(model, val_loader, loss_fn, device)
    if abs(checked_loss - saved["val_loss"]) > 1e-5 or checked_accuracy != saved["val_accuracy"]:
        raise RuntimeError("Geri yuklenen modelin dogrulama sonucu kayitla uyusmuyor.")
    plot_history(history, name, run_dir / "learning_curves.png")
    result = {
        "run_name": name, "parent_run": parent, "candidate_config": config,
        "best_epoch": saved["epoch"], "val_loss": checked_loss,
        "val_accuracy": checked_accuracy, "parameters": run_config["parameters"],
        "training_and_validation_seconds": sum(row["epoch_seconds"] for row in history),
        "checkpoint": (Path(name) / "best.pt").as_posix(),
    }
    save_json(run_dir / "summary.json", result)
    return result, history


def run_improvements(project_dir, output_dir, epochs=30):
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    if epochs < 1:
        raise ValueError("Epoch sayisi en az 1 olmali.")
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = json.loads((project_dir / "results" / "experiment" / "summary.json").read_text(encoding="utf-8"))
    reference = next(row for row in v1["results"] if row["model_name"] == "cnn")
    split_path = project_dir / "results" / "split_indices.npz"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != v1["config"]["split_sha256"]:
        raise ValueError("Bolme V1 deneyindeki dosyayla ayni olmali.")
    with np.load(split_path, allow_pickle=False) as split:
        train_indices, val_indices = split["train_indices"], split["val_indices"]
        seed = int(split["seed"].item())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stages = [
        ("01_longer", {}),
        ("02_scheduler", {"scheduler": True}),
        ("03_max_pool", {"pooling": "max"}),
        ("04_wider", {"channels": [16, 32]}),
        ("05_augmentation", {"augmentation": True}),
    ]
    protocol = {
        "version": "V2 validation-only improvement study", "seed": seed,
        "epochs_per_candidate": epochs, "split_sha256": split_hash,
        "dataset": "Fashion-MNIST", "train_samples": len(train_indices),
        "val_samples": len(val_indices), "batch_size": 64,
        "optimizer": "Adam", "initial_lr": 0.001, "loss": "CrossEntropyLoss",
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "stages": [{"name": name, "change": change} for name, change in stages],
        "selection": "highest validation accuracy; tie lower validation loss",
        "design": "greedy: each change is applied to the best preceding candidate",
        "v1_validation_reference": reference,
        "test_policy": "V1 test already seen. This script never loads test data. Later benchmark reuse must be disclosed.",
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "numpy": str(np.__version__),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    save_json(output_dir / "protocol.json", protocol)
    winning_config = {"channels": [6, 16], "pooling": "avg",
                      "scheduler": False, "augmentation": False}
    winner, results, histories = None, [], {}
    print(f"V2: {len(stages)} CNN adayi x {epochs} epoch | cihaz={device}", flush=True)
    print("Secim sadece dogrulama ile; resmi test bolumu kullanilmaz.", flush=True)
    for name, changes in stages:
        candidate_config = winning_config.copy()
        candidate_config.update(changes)
        parent = winner["run_name"] if winner else "V1 architecture; fresh weights"
        print(f"\n{name} | parent={parent} | ayarlar={candidate_config}", flush=True)
        result, history = train_candidate(
            project_dir, output_dir / name, name, parent, candidate_config,
            train_indices, val_indices, seed, epochs, device,
        )
        results.append(result)
        histories[name] = history
        if winner is None or better(result["val_accuracy"], result["val_loss"],
                                    winner["val_accuracy"], winner["val_loss"]):
            winner, winning_config = result, candidate_config.copy()
        print(f"Simdiki en iyi aday: {winner['run_name']} | val={winner['val_accuracy']:.2f}%", flush=True)

    # Kazanan icin sinif metrikleri ve yanlis ornekler; yine yalnizca dogrulama.
    selected = torch.load(output_dir / winner["checkpoint"], map_location=device, weights_only=True)
    selected_model = build_cnn(channels=selected["candidate_config"]["channels"], pooling=selected["candidate_config"]["pooling"]).to(device)
    selected_model.load_state_dict(selected["model_state_dict"])
    val_source = FashionMNIST(project_dir / "data", train=True, download=False, transform=ToTensor())
    winner_analysis = analyze_split(
        val_source, val_indices, {"cnn": selected_model}, device, 64,
        output_dir / "winner_validation", "validation", seed,
    )["cnn"]
    plot_comparison(histories, output_dir / "validation_comparison.png")
    ranking = sorted(results, key=lambda row: (-row["val_accuracy"], row["val_loss"]))
    with (output_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["run_name", "parent_run", "best_epoch", "val_accuracy", "val_loss", "parameters"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranking)
    improved = better(winner["val_accuracy"], winner["val_loss"],
                      reference["val_accuracy"], reference["val_loss"])
    report = {
        "protocol": protocol, "results": results, "winner": winner,
        "winner_validation_metrics": winner_analysis,
        "improved_over_v1_validation": improved,
        "validation_gain_percentage_points": winner["val_accuracy"] - reference["val_accuracy"],
    }
    save_json(output_dir / "summary.json", report)
    print("\nV2 DOGRULAMA OZETI", flush=True)
    for row in ranking:
        print(f"{row['run_name']}: epoch={row['best_epoch']} | "
              f"val acc={row['val_accuracy']:.2f}% | loss={row['val_loss']:.4f}", flush=True)
    print(f"Kazanan: {winner['run_name']} | V1'e fark={report['validation_gain_percentage_points']:+.2f} yuzde puani", flush=True)
    print(f"V1 dogrulamasini gecti: {improved} | macro-F1={winner_analysis['macro_f1']:.4f}", flush=True)
    print(f"Kayitlar: {output_dir}", flush=True)
    return report

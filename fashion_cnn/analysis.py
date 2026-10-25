"""Frozen V1 evaluation and reusable report generation."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from .artifacts import file_hash, save_json
from .engine import collect_predictions
from .metrics import classification_metrics
from .models import build_baseline, build_cnn
from .plotting import plot_confusion, save_gallery


def analyze_split(raw_dataset, source_indices, models, device, batch_size,
                  output_dir, split_name, seed):
    """Ayni veri bolumunu iki secilmis modelle olc; rapor ve gorselleri yaz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = raw_dataset.classes
    dataset = Subset(raw_dataset, source_indices.tolist())
    summary = {}
    for name, model in models.items():
        model_dir = output_dir / name
        model_dir.mkdir(exist_ok=True)
        targets, predictions, probabilities, mean_loss = collect_predictions(
            model, dataset, device, batch_size
        )
        matrix, metrics, per_class = classification_metrics(targets, predictions, classes)
        metrics.update({"loss": mean_loss, "model_name": name, "split": split_name})
        summary[name] = metrics
        save_json(model_dir / "metrics.json", metrics)
        with (model_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_class[0]))
            writer.writeheader()
            writer.writerows(per_class)
        with (model_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["true/predicted", *classes])
            for label, row in zip(classes, matrix):
                writer.writerow([label, *row.tolist()])
        for normalized, filename in ((False, "confusion_counts.png"), (True, "confusion_normalized.png")):
            plot_confusion(matrix, classes, name, normalized, model_dir / filename)

        fields = ["split", "dataset_index", "true_label", "true_class", "predicted_label",
                  "predicted_class", "correct", "softmax_score", *[f"p_{i}" for i in range(len(classes))]]
        with (model_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            for index, true, predicted, scores in zip(source_indices, targets, predictions, probabilities):
                writer.writerow([split_name, int(index), int(true), classes[int(true)], int(predicted),
                                 classes[int(predicted)], int(true == predicted), float(scores[predicted]),
                                 *scores.tolist()])

        errors = [(int(matrix[i, j]), i, j) for i in range(len(classes))
                  for j in range(len(classes)) if i != j and matrix[i, j] > 0]
        errors.sort(key=lambda item: (-item[0], item[1], item[2]))
        with (model_dir / "top_confusions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["true_class", "predicted_class", "count"])
            writer.writerows((classes[i], classes[j], count) for count, i, j in errors)

        if name == "cnn":
            wrong = np.flatnonzero(targets != predictions)
            chosen_scores = probabilities[np.arange(len(predictions)), predictions]
            random_positions = np.random.default_rng(seed).choice(wrong, size=min(10, len(wrong)), replace=False)
            top_positions = wrong[np.argsort(-chosen_scores[wrong], kind="stable")[:10]]
            random_entries = save_gallery(
                raw_dataset, source_indices, targets, predictions, probabilities, random_positions,
                f"CNN: random errors ({split_name})", model_dir / "random_errors.png",
            )
            top_entries = save_gallery(
                raw_dataset, source_indices, targets, predictions, probabilities, top_positions,
                f"CNN: errors with highest softmax scores ({split_name})", model_dir / "high_score_errors.png",
            )
            overlap = sorted(set(int(source_indices[i]) for i in random_positions)
                             & set(int(source_indices[i]) for i in top_positions))
            save_json(model_dir / "error_gallery_indices.json", {
                "seed": seed, "random_errors": random_entries, "high_score_errors": top_entries,
                "overlapping_dataset_indices": overlap,
                "score_note": "Maximum softmax score is not a calibrated correctness guarantee.",
            })
        print(f"{split_name} | {name}: accuracy={metrics['accuracy_pct']:.2f}% | "
              f"macro-F1={metrics['macro_f1']:.4f} | errors={metrics['errors']}", flush=True)
    save_json(output_dir / "summary.json", summary)
    return summary


def run_analysis(project_dir, output_dir, include_final_test=False):
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = project_dir / "results" / "experiment"
    training_report = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
    config = training_report["config"]
    split_path = project_dir / "results" / "split_indices.npz"
    if file_hash(split_path) != config["split_sha256"]:
        raise ValueError("Veri bolmesi egitimde kullanilan dosyayla eslesmiyor.")
    with np.load(split_path, allow_pickle=False) as split:
        seed, val_indices = int(split["seed"].item()), split["val_indices"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    builders = {"linear_baseline": build_baseline, "cnn": build_cnn}
    models, checkpoint_records = {}, {}
    for record in training_report["results"]:
        name = record["model_name"]
        checkpoint_path = experiment_dir / record["checkpoint"]
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if saved["epoch"] != record["best_epoch"] or saved["seed"] != seed:
            raise ValueError(f"{name}: checkpoint ve egitim ozeti eslesmiyor.")
        model = builders[name]().to(device)
        model.load_state_dict(saved["model_state_dict"])
        models[name] = model
        checkpoint_records[name] = {"sha256": file_hash(checkpoint_path), "epoch": saved["epoch"]}

    print("SECILEN MODELLER: egitim yapilmadan degerlendiriliyor.", flush=True)
    train_source = FashionMNIST(project_dir / "data", train=True, download=False, transform=ToTensor())
    validation = analyze_split(train_source, val_indices, models, device, config["batch_size"],
                               output_dir / "validation", "validation", seed)
    for record in training_report["results"]:
        result = validation[record["model_name"]]
        if abs(result["loss"] - record["val_loss"]) > 1e-5 or abs(result["accuracy_pct"] - record["val_accuracy"]) > 1e-5:
            raise ValueError("Dogrulama sonuclari egitim kaydiyla uyusmuyor; test acilmadi.")
    if not include_final_test:
        print("Yalnizca dogrulama analizi tamamlandi; resmi test bolumu acilmadi.", flush=True)
        return {"validation": validation}

    # Test goruntuleri acilmadan once model ve olcum kurallari kayda alinir.
    frozen_path = output_dir / "frozen_protocol.json"
    identity = {"checkpoints": checkpoint_records, "split_sha256": config["split_sha256"]}
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen["identity"] != identity:
            raise ValueError("Bu nihai test kaydi farkli model agirliklarina ait; deneyleri karistirmayin.")
    else:
        save_json(frozen_path, {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
            "training_config": config, "test_samples": 10000,
            "metrics": ["accuracy_pct", "macro_f1", "per_class_precision_recall_f1_support"],
            "metric_definition": "Rows=true, columns=predicted; all 10 labels; zero_division=0.",
            "preprocessing": "ToTensor: grayscale 1x28x28, float32, pixels / 255.",
            "test_policy": "Frozen checkpoints only; no model fitting or tuning from test results.",
        })
    test_summary_path = output_dir / "test" / "summary.json"
    if test_summary_path.exists():
        test = json.loads(test_summary_path.read_text(encoding="utf-8"))
        print("Mevcut nihai test raporu okundu; test tahminleri yeniden uretilmedi.", flush=True)
    else:
        test_source = FashionMNIST(project_dir / "data", train=False, download=False, transform=ToTensor())
        test_indices = np.arange(len(test_source), dtype=np.int64)
        test = analyze_split(test_source, test_indices, models, device, config["batch_size"],
                             output_dir / "test", "test", seed)
    report = {"validation": validation, "test": test, "protocol": "frozen_protocol.json"}
    save_json(output_dir / "summary.json", report)
    print("\nNIHAI TEST OZETI", flush=True)
    for name, result in test.items():
        print(f"{name}: accuracy={result['accuracy_pct']:.2f}% | "
              f"macro-F1={result['macro_f1']:.4f} | errors={result['errors']}/{result['samples']}", flush=True)
    print(f"Analiz dosyalari: {output_dir}", flush=True)
    return report

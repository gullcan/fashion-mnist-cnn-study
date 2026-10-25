"""Evaluate the fixed V2 artifact; test reuse is disclosed and never used for tuning."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from ..analysis import analyze_split
from ..artifacts import file_hash, save_json
from ..models import build_cnn


def run_evaluation(project_dir, output_dir, include_test=False):
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    v2_dir = project_dir / "results" / "improvement_v2"
    training = json.loads((v2_dir / "summary.json").read_text(encoding="utf-8"))
    stability = json.loads((project_dir / "results" / "stability_v2" / "summary.json").read_text(encoding="utf-8"))
    winner = training["winner"]
    checkpoint_path = v2_dir / winner["checkpoint"]
    fixed = stability["protocol"]
    if file_hash(checkpoint_path) != fixed["fixed_checkpoint_sha256"]:
        raise ValueError("Model, kararlilik kontrolunde sabitlenen dosyayla ayni degil.")
    if winner["candidate_config"] != fixed["candidate_config"]:
        raise ValueError("Model ayarlari sabitlenen protokolle uyusmuyor.")
    split_path = project_dir / "results" / "split_indices.npz"
    if file_hash(split_path) != training["protocol"]["split_sha256"]:
        raise ValueError("Veri bolmesi V2 egitimindeki dosyayla ayni degil.")
    with np.load(split_path, allow_pickle=False) as split:
        val_indices = split["val_indices"]
    seed = int(training["protocol"]["seed"])
    if seed != 42:
        raise ValueError("Bu asamada onceden sabitlenen seed42 modeli kullanilir.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if (saved["seed"] != seed or saved["epoch"] != winner["best_epoch"]
            or saved["candidate_config"] != winner["candidate_config"]
            or saved["run_name"] != winner["run_name"]):
        raise ValueError("Checkpoint ile V2 ozeti uyusmuyor.")
    model = build_cnn(channels=saved["candidate_config"]["channels"], pooling=saved["candidate_config"]["pooling"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    train_source = FashionMNIST(project_dir / "data", train=True,
                               download=False, transform=ToTensor())
    if list(train_source.classes) != saved["classes"]:
        raise ValueError("Sinif sirasi egitimdeki sirayla uyusmuyor.")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("V2: seed42 checkpoint yuklendi; egitim yapilmiyor.", flush=True)
    validation = analyze_split(
        train_source, val_indices, {"cnn": model}, device, 64,
        output_dir / "validation", "validation", seed,
    )["cnn"]
    if (abs(validation["accuracy_pct"] - winner["val_accuracy"]) > 1e-5
            or abs(validation["loss"] - winner["val_loss"]) > 1e-5):
        raise ValueError("Dogrulama sonucu kayitla uyusmuyor; test acilmadi.")
    if not include_test:
        print("Yalnizca dogrulama kontrolu tamamlandi; test acilmadi.", flush=True)
        return {"validation": validation}

    v1_path = project_dir / "results" / "analysis" / "test" / "summary.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))["cnn"]
    identity = {
        "checkpoint_sha256": file_hash(checkpoint_path),
        "split_sha256": file_hash(split_path),
        "v1_test_summary_sha256": file_hash(v1_path),
        "candidate_config": winner["candidate_config"], "seed": seed,
        "epoch": saved["epoch"],
    }
    # Resmi test dataset'i kurulmadan once bu surum ve olcum kurallari sabitlenir.
    frozen_path = output_dir / "frozen_protocol.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen["identity"] != identity:
            raise ValueError("Mevcut V2 test raporu baska modele/protokole ait.")
    else:
        frozen = {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity, "test_samples": 10000,
            "classes": saved["classes"],
            "preprocessing": "ToTensor only: grayscale 1x28x28, float32, pixels / 255.",
            "metrics": "Accuracy, macro-F1 and all ten per-class precision/recall/F1; zero_division=0.",
            "test_policy": "Official benchmark reused after V1. No fitting, selection or tuning from this evaluation.",
            "seed_policy": "Seed42 was fixed before stability results; other seeds are not test candidates.",
            "selection_caveat": "V2 settings were selected on seed42 validation. Stability mean is descriptive, not an unbiased estimate or confidence interval.",
            "stability": {key: stability[key] for key in (
                "mean_val_accuracy_pct", "sample_std_percentage_points",
            )},
            "torch": str(torch.__version__), "device": str(device),
        }
        save_json(frozen_path, frozen)
    test_summary_path = output_dir / "test" / "summary.json"
    if test_summary_path.exists():
        test = json.loads(test_summary_path.read_text(encoding="utf-8"))["cnn"]
        print("Kayitli V2 test raporu okundu; test tahminleri yeniden uretilmedi.", flush=True)
    else:
        test_source = FashionMNIST(project_dir / "data", train=False,
                                  download=False, transform=ToTensor())
        if len(test_source) != 10000 or list(test_source.classes) != saved["classes"]:
            raise ValueError("Resmi test boyutu veya sinif sirasi beklenenden farkli.")
        test = analyze_split(
            test_source, np.arange(len(test_source), dtype=np.int64),
            {"cnn": model}, device, 64, output_dir / "test", "test", seed,
        )["cnn"]
    if test["samples"] != 10000 or v1["samples"] != 10000:
        raise ValueError("Test karsilastirmasi ayni resmi 10.000 ornegi kullanmali.")
    comparison = {
        "accuracy_gain_percentage_points": test["accuracy_pct"] - v1["accuracy_pct"],
        "fewer_errors": v1["errors"] - test["errors"],
        "macro_f1_gain": test["macro_f1"] - v1["macro_f1"],
    }
    report = {"validation": validation, "test": test, "v1_cnn_test": v1,
              "comparison": comparison, "protocol": "frozen_protocol.json"}
    save_json(output_dir / "summary.json", report)
    with (output_dir / "v1_v2_test_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["version", "accuracy_pct", "macro_f1", "errors", "samples"])
        writer.writeheader()
        for version, result in (("V1", v1), ("V2_seed42", test)):
            writer.writerow({"version": version, **{key: result[key] for key in writer.fieldnames[1:]}})
    print("\nV2 TEST OZETI", flush=True)
    print(f"V1 CNN: accuracy={v1['accuracy_pct']:.2f}% | macro-F1={v1['macro_f1']:.4f} | errors={v1['errors']}/10000", flush=True)
    print(f"V2 CNN: accuracy={test['accuracy_pct']:.2f}% | macro-F1={test['macro_f1']:.4f} | errors={test['errors']}/10000", flush=True)
    print(f"Fark: {comparison['accuracy_gain_percentage_points']:+.2f} yuzde puani | "
          f"daha az hata={comparison['fewer_errors']}", flush=True)
    print("Ayni resmi test yeniden kullanildi; yeni kor test degildir.", flush=True)
    print(f"Kayitlar: {output_dir}", flush=True)
    return report

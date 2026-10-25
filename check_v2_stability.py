"""Secilen V2 ayarlarini iki ek seed ile egit; test verisini acma.

Seed42 kazanan dosyasi sabit kalir. Ek seed'lerden yeni kazanan secilmez.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from improve_cnn import save_json, train_candidate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_stability(project_dir, output_dir):
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    source = project_dir / "results" / "improvement_v2"
    report = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    winner, source_protocol = report["winner"], report["protocol"]
    seed42 = int(source_protocol["seed"])
    if seed42 != 42:
        raise ValueError("Bu kontrol mevcut seed42 V2 sonucuna gore tasarlandi.")
    epochs = int(source_protocol["epochs_per_candidate"])
    checkpoint = source / winner["checkpoint"]
    checkpoint_hash = sha256(checkpoint)
    split_path = project_dir / "results" / "split_indices.npz"
    if sha256(split_path) != source_protocol["split_sha256"]:
        raise ValueError("V2 ile ayni bolme dosyasi kullanilmali.")
    with np.load(split_path, allow_pickle=False) as split:
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    protocol = {
        "source_run": winner["run_name"],
        "candidate_config": winner["candidate_config"],
        "epochs": epochs, "seeds": [42, 43, 44],
        "split_sha256": source_protocol["split_sha256"],
        "fixed_deployment_checkpoint": str(checkpoint.resolve()),
        "fixed_checkpoint_sha256": checkpoint_hash,
        "selection": "Same V2 rule within each run. Never select a seed by its score.",
        "test_policy": "Validation only; no official test dataset is loaded.",
        "interpretation": "Descriptive variation on one fixed split; three runs are not a confidence interval.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "protocol.json", protocol)
    rows = [{"seed": 42, "best_epoch": winner["best_epoch"],
             "val_accuracy": winner["val_accuracy"], "val_loss": winner["val_loss"],
             "origin": "existing V2 selected checkpoint"}]
    print(f"Secilen ayarlar: {winner['candidate_config']}", flush=True)
    print(f"Seed42 yeniden egitilmez; seed43 ve seed44 x {epochs} epoch.", flush=True)
    print("Bolme sabit; degisenler baslangic agirliklari ve egitim rastgeleligi.", flush=True)
    for seed in (43, 44):
        name = f"seed_{seed}"
        result, _ = train_candidate(
            project_dir, output_dir / name, name, winner["run_name"],
            winner["candidate_config"], train_indices, val_indices,
            seed, epochs, device,
        )
        rows.append({"seed": seed, "best_epoch": result["best_epoch"],
                     "val_accuracy": result["val_accuracy"],
                     "val_loss": result["val_loss"], "origin": name})
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Sabit V2 modelinin dosyasi degisti.")
    accuracies = np.array([row["val_accuracy"] for row in rows])
    summary = {
        "protocol": protocol, "runs": rows,
        "mean_val_accuracy_pct": float(accuracies.mean()),
        "sample_std_percentage_points": float(accuracies.std(ddof=1)),
        "min_val_accuracy_pct": float(accuracies.min()),
        "max_val_accuracy_pct": float(accuracies.max()),
    }
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "seed_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nV2 KARARLILIK OZETI", flush=True)
    for row in rows:
        print(f"seed={row['seed']} | epoch={row['best_epoch']} | "
              f"val acc={row['val_accuracy']:.2f}% | loss={row['val_loss']:.4f}", flush=True)
    print(f"Ortalama: {summary['mean_val_accuracy_pct']:.2f}% | "
          f"standart sapma: {summary['sample_std_percentage_points']:.3f} yuzde puani", flush=True)
    print("Bu olcum farkli veri bolmelerini veya yeni fotograflari kapsamaz.", flush=True)
    print("Sunulacak checkpoint seed42 olarak sabit; en sansli seed secilmedi.", flush=True)
    print(f"Kayitlar: {output_dir}", flush=True)
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    run_stability(root, root / "results" / "stability_v2")

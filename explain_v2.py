"""Sabit V2 modeli icin filtre/aktivasyon gorselleri ve eslesmis hata analizi.

Egitim yoktur. Gorsellestirme ornekleri yalnizca dogrulamadan gelir.
Test tahminleri mevcut CSV'lerden okunur; test dataset'i acilmaz.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from analyze_experiment import file_hash, save_json
from improve_cnn import build_candidate


def compare_errors(project_dir, output_dir):
    def read_predictions(path):
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        records = {int(row["dataset_index"]): row for row in rows}
        if len(records) != len(rows) or len(rows) != 10000:
            raise ValueError("Her test goruntusu tam bir kez bulunmali.")
        for row in rows:
            if int(row["correct"]) != int(row["true_label"] == row["predicted_label"]):
                raise ValueError("CSV dogruluk alani etiketlerle uyusmuyor.")
        return records
    paths = {
        "v1": project_dir / "results" / "analysis" / "test" / "cnn" / "predictions.csv",
        "v2": project_dir / "results" / "evaluation_v2" / "test" / "cnn" / "predictions.csv",
    }
    old, new = read_predictions(paths["v1"]), read_predictions(paths["v2"])
    if set(old) != set(new):
        raise ValueError("Karsilastirilan test indeksleri ayni olmali.")
    counts = dict.fromkeys(["both_correct", "fixed_by_v2", "broken_by_v2", "both_wrong"], 0)
    with (output_dir / "error_transitions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset_index", "true_class", "v1_predicted_class", "v2_predicted_class", "transition"])
        for index in sorted(old):
            a, b = old[index], new[index]
            if a["true_label"] != b["true_label"]:
                raise ValueError("Ayni indeks icin gercek etiketler uyusmuyor.")
            c1, c2 = int(a["correct"]), int(b["correct"])
            transition = ("both_correct" if c1 and c2 else "fixed_by_v2" if c2
                          else "broken_by_v2" if c1 else "both_wrong")
            counts[transition] += 1
            writer.writerow([index, b["true_class"], a["predicted_class"], b["predicted_class"], transition])
    with (project_dir / "results" / "evaluation_v2" / "summary.json").open(encoding="utf-8") as file:
        evaluation = json.load(file)
    if (counts["fixed_by_v2"] + counts["both_wrong"] != evaluation["v1_cnn_test"]["errors"]
            or counts["broken_by_v2"] + counts["both_wrong"] != evaluation["test"]["errors"]):
        raise ValueError("Eslesmis hata sayilari nihai raporlarla uyusmuyor.")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.bar(["Both correct", "V1 wrong / V2 correct", "V1 correct / V2 wrong", "Both wrong"],
           list(counts.values()), color=["#4c78a8", "#59a14f", "#e15759", "#9c755f"])
    ax.bar_label(ax.containers[0], padding=3)
    ax.set(ylabel="Test images", title="Same 10,000 images: V1 and V2 outcomes", ylim=(0, 10000))
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(output_dir / "error_transitions.png", dpi=150)
    plt.close(fig)
    return {"counts": counts, "input_sha256": {key: file_hash(path) for key, path in paths.items()}}


def plot_filters(initial, trained, output_dir):
    limit = max(float(np.abs(initial).max()), float(np.abs(trained).max()), 1e-12)
    fig, axes = plt.subplots(4, 8, figsize=(14, 8), layout="constrained")
    for offset, weights, label in ((0, initial, "Initial"), (16, trained, "Trained")):
        for channel, weight in enumerate(weights):
            ax = axes.ravel()[offset + channel]
            image = ax.imshow(weight[0], cmap="coolwarm", vmin=-limit, vmax=limit,
                              interpolation="nearest")
            ax.set_title(f"{label}: {channel}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Weight; same scale in all panels")
    fig.suptitle("First convolution: 16 filters, each 1 x 5 x 5\nTop: reconstructed seed42 initialization; bottom: trained weights")
    fig.savefig(output_dir / "filters_before_after.png", dpi=150)
    plt.close(fig)
    np.save(output_dir / "filters_initial_reconstructed.npy", initial)
    np.save(output_dir / "filters_trained.npy", trained)
    return limit


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


def explain(project_dir, output_dir):
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    v2_dir = project_dir / "results" / "improvement_v2"
    report = json.loads((v2_dir / "summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((project_dir / "results" / "evaluation_v2" / "frozen_protocol.json").read_text(encoding="utf-8"))
    checkpoint = v2_dir / report["winner"]["checkpoint"]
    before_hash = file_hash(checkpoint)
    if before_hash != frozen["identity"]["checkpoint_sha256"]:
        raise ValueError("Aciklanan model, test edilen sabit model olmali.")
    split_path = project_dir / "results" / "split_indices.npz"
    if file_hash(split_path) != frozen["identity"]["split_sha256"]:
        raise ValueError("Dogrulama bolmesi degisti.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["candidate_config"]
    if config["channels"] != [16, 32]:
        raise ValueError("Bu gorsel duzeni secilen 16/32 kanalli V2 icindir.")
    if str(torch.__version__) != report["protocol"]["torch"]:
        raise ValueError("Baslangic filtrelerini yeniden kurmak icin ayni PyTorch surumunu kullanin.")
    # Bunlar egitim aninda kaydedilmis V2 filtreleri degil: ayni seed/factory ile yeniden kurulur.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(saved["seed"])
        initial = build_candidate(config)[0].weight.detach().numpy().copy()
    model = build_candidate(config)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    trained = model[0].weight.detach().numpy().copy()
    filter_limit = plot_filters(initial, trained, output_dir)
    source = FashionMNIST(project_dir / "data", train=True, download=False, transform=ToTensor())
    with np.load(split_path, allow_pickle=False) as split:
        indices = split["val_indices"]
    chosen = [int(indices[np.flatnonzero(source.targets[indices].numpy() == label)[0]]) for label in (0, 6, 9)]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout="constrained")
    examples = []
    for row, index in enumerate(chosen):
        image, target = source[index]
        activations = {}
        with torch.no_grad():
            value = image.unsqueeze(0)
            for layer_id, layer in enumerate(model):
                value = layer(value)
                if layer_id in (0, 1, 2, 4):
                    activations[layer_id] = value[0].numpy().copy()
            predicted = int(value.argmax(dim=1).item())
        raw, relu, pooled = (activations[layer_id][0] for layer_id in (0, 1, 2))
        limit = max(float(np.abs(raw).max()), 1e-12)
        for col, (values, title) in enumerate(((image[0].numpy(), "Input 28x28"),
                (raw, "Conv1 channel0 24x24"), (relu, "ReLU channel0 24x24"),
                (pooled, "AvgPool channel0 12x12"))):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(values, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                heatmap = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"ID {index}\nTrue: {source.classes[target]}\nPred: {source.classes[predicted]}", fontsize=9)
        fig.colorbar(heatmap, ax=axes[row, 1:].tolist(), shrink=0.7)
        examples.append({"validation_source_index": index, "true_label": target,
                         "predicted_label": predicted, "conv1_channel0_scale": [-limit, limit]})
        if target == 6:
            plot_channels(activations[1], f"Shirt ID {index}: Conv1 after ReLU (16 x 24 x 24)",
                          4, output_dir / "shirt_conv1_maps.png")
            plot_channels(activations[4], f"Shirt ID {index}: Conv2 after ReLU (32 x 8 x 8)",
                          8, output_dir / "shirt_conv2_maps.png")
            np.savez(output_dir / "shirt_activations.npz", conv1_relu=activations[1], conv2_relu=activations[4])
    fig.suptitle("Same first filter: convolution -> ReLU -> average pooling\nShared color scale within each row; zero is neutral")
    fig.savefig(output_dir / "activation_flow.png", dpi=150)
    plt.close(fig)
    transitions = compare_errors(project_dir, output_dir)
    if file_hash(checkpoint) != before_hash:
        raise RuntimeError("Sabit checkpoint dosyasi degisti.")
    summary = {
        "checkpoint_sha256": before_hash, "examples": examples, "error_transitions": transitions,
        "filter_plot_scale": [-filter_limit, filter_limit],
        "initialization_note": "Reconstructed from seed and identical factory/version, not captured during V2 training.",
        "interpretation": "Feature maps are responses, not proof of the reason for a prediction. Channel0 is illustrative, not selected as most important.",
        "sample_rule": "First saved validation index in each of classes 0,6,9; not selected by correctness.",
    }
    save_json(output_dir / "summary.json", summary)
    print("\nV2 MEKANIZMA VE HATA OZETI", flush=True)
    print(f"Ilk katman: {tuple(trained.shape)} | ornek indeksleri: {chosen}", flush=True)
    counts = transitions["counts"]
    print(f"V2'nin duzelttigi: {counts['fixed_by_v2']} | V2'nin bozdugu: {counts['broken_by_v2']} | "
          f"ikisinin de yanildigi: {counts['both_wrong']}", flush=True)
    print("Egitim yapilmadi; test dataset'i acilmadi; model dosyasi korundu.", flush=True)
    print(f"Gorseller: {output_dir}", flush=True)
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    explain(root, root / "results" / "explanation_v2")

"""Export the fixed trained CNN and validation-only parity fixtures for the browser.

No training occurs. The official test dataset is never opened. Reference logits
come from the real PyTorch model, independently of the JavaScript implementation.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

EXPECTED_CHECKPOINT = "013745dcd66e4d5c462ba64607faed70831ea7e9caa27721afcdc34693b245f2"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def export(project, output, references):
    project, output, references = Path(project).resolve(), Path(output).resolve(), Path(references).resolve()
    checkpoint = project / "results/improvement_v2/05_augmentation/best.pt"
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if original_hash != EXPECTED_CHECKPOINT:
        raise ValueError("The fixed evaluated checkpoint does not match its recorded identity.")
    sys.path.insert(0, str(project))
    from fashion_cnn.models import build_cnn

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_config = {"channels": [16, 32], "pooling": "avg", "scheduler": False, "augmentation": True}
    if saved["candidate_config"] != expected_config or saved["seed"] != 42 or saved["epoch"] != 30:
        raise ValueError("Unexpected architecture or selected checkpoint.")
    model = build_cnn(channels=saved["candidate_config"]["channels"], pooling=saved["candidate_config"]["pooling"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    torch.set_num_threads(1)
    model_dir, example_dir = output / "model", output / "examples"
    for directory in (model_dir, example_dir, references):
        directory.mkdir(parents=True, exist_ok=True)

    chunks, tensors, offset = [], {}, 0
    for name, parameter in model.state_dict().items():
        array = parameter.detach().cpu().numpy().astype("<f4", copy=False)
        tensors[name] = {"shape": list(array.shape), "offset": offset, "length": int(array.size)}
        chunks.append(array.tobytes(order="C"))
        offset += int(array.size)
    weights = b"".join(chunks)
    if offset != 57190:
        raise ValueError("Unexpected parameter count.")
    (model_dir / "weights.bin").write_bytes(weights)
    evaluation = json.loads((project / "results/evaluation_v2/summary.json").read_text(encoding="utf-8"))
    metadata = {
        "format": "cnn-learning-browser-v1", "dtype": "float32-le",
        "layout": "NCHW; convolution OIHW; linear output-input",
        "classes": saved["classes"], "parameterCount": offset,
        "checkpointSha256": original_hash,
        "weightsFile": "weights.bin", "weightsSha256": hashlib.sha256(weights).hexdigest(),
        "weightsBytes": len(weights), "inputShape": [1, 28, 28],
        "tensors": tensors,
        "architecture": "Conv1(1->16,k5),ReLU,AvgPool2,Conv2(16->32,k5),ReLU,AvgPool2,Flatten512,Linear84,ReLU,Linear10",
        "training": {"seed": 42, "selectedEpoch": 30, "optimizer": "Adam", "learningRate": 0.001,
                     "augmentation": ["RandomCrop(28,padding=2)", "RandomHorizontalFlip(0.5)"],
                     "trainSamples": 54000, "validationSamples": 6000},
        "metrics": {"testAccuracyPct": evaluation["test"]["accuracy_pct"],
                    "testMacroF1": evaluation["test"]["macro_f1"],
                    "validationAccuracyPct": evaluation["validation"]["accuracy_pct"]},
        "evaluationNote": "The official Fashion-MNIST test was reused after V1; arbitrary uploaded-photo accuracy has not been evaluated.",
        "preprocessingNote": "Native dataset inference uses grayscale 28x28 float32 values in [0,1]. Uploaded-photo preparation is a separate heuristic.",
    }
    write_json(model_dir / "model.json", metadata)

    dataset = FashionMNIST(project / "data", train=True, download=False, transform=ToTensor())
    if list(dataset.classes) != saved["classes"]:
        raise ValueError("Dataset class ordering differs from the checkpoint.")
    with np.load(project / "results/split_indices.npz", allow_pickle=False) as split:
        validation = split["val_indices"]
    labels = dataset.targets.numpy()
    examples, parity_indices = [], []
    for label in range(10):
        members = validation[labels[validation] == label]
        index = int(members[0])
        filename = f"example-{label}.png"
        Image.fromarray(dataset.data[index].numpy()).save(example_dir / filename)
        examples.append({"id": f"fashion-{label}", "index": index, "labelIndex": label,
                         "label": dataset.classes[label], "name": dataset.classes[label],
                         "image": f"./examples/{filename}"})
        parity_indices.extend(int(value) for value in members[:10])
    write_json(example_dir / "examples.json", examples)
    if (project / "third_party/FASHION_MNIST_LICENSE.txt").is_file():
        shutil.copy2(project / "third_party/FASHION_MNIST_LICENSE.txt", example_dir / "LICENSE.txt")

    inputs = torch.stack([dataset[index][0] for index in parity_indices])
    with torch.no_grad():
        logits = model(inputs)
        probabilities = logits.softmax(dim=1)
        value = inputs[:1]
        activations = []
        names = {0: "conv1", 1: "relu1", 2: "pool1", 3: "conv2", 4: "relu2", 5: "pool2"}
        activation_chunks, activation_offset = [], 0
        for layer_id, layer in enumerate(model):
            value = layer(value)
            if layer_id in names:
                array = value[0].numpy().astype("<f4", copy=False)
                activations.append({"name": names[layer_id], "shape": list(array.shape),
                                    "offset": activation_offset, "length": int(array.size)})
                activation_chunks.append(array.tobytes(order="C"))
                activation_offset += int(array.size)
    (references / "runtime-inputs.bin").write_bytes(inputs.numpy().astype("<f4", copy=False).tobytes(order="C"))
    (references / "runtime-activations.bin").write_bytes(b"".join(activation_chunks))
    expected = {"source": "PyTorch CPU, fixed model; 10 first saved validation indices per class",
                "torch": str(torch.__version__), "indices": parity_indices,
                "logits": logits.numpy().tolist(), "probabilities": probabilities.numpy().tolist(),
                "top1": logits.argmax(dim=1).tolist(), "activationChecks": activations,
                "checkpointSha256": original_hash}
    write_json(references / "runtime-expected.json", expected)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("Source checkpoint changed unexpectedly.")
    print(f"Exported {offset} parameters ({len(weights)} bytes), {len(examples)} examples and {len(parity_indices)} independent parity inputs.")
    print(f"Original checkpoint preserved: {original_hash}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "dist")
    parser.add_argument("--references", type=Path, default=Path(__file__).resolve().parent / "references")
    args = parser.parse_args()
    export(args.project, args.out, args.references)

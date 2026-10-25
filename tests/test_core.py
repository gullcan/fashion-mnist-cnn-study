"""Behavioral regression checks; no downloads, GPU or training campaign required."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fashion_cnn.artifacts import file_hash, save_json
from fashion_cnn.cli import fresh_output, make_parser
from fashion_cnn.data import load_split, make_loaders, stratified_split
from fashion_cnn.engine import collect_predictions, evaluate, train_one_epoch
from fashion_cnn.inference import load_checkpoint, load_selected_model, predict_tensor
from fashion_cnn.metrics import classification_metrics
from fashion_cnn.models import build_baseline, build_cnn

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def test_recorded_state_dicts_still_load(self):
        for model, path, parameters in (
            (build_baseline(), "results/experiment/linear_baseline/best.pt", 7850),
            (build_cnn(), "results/experiment/cnn/best.pt", 25010),
            (build_cnn((16, 32)), "results/improvement_v2/05_augmentation/best.pt", 57190),
        ):
            saved = torch.load(ROOT / path, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["model_state_dict"], strict=True)
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.zeros(2, 1, 28, 28)).shape), (2, 10))
            self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)

    def test_reject_invalid_architecture(self):
        for channels, pooling in (((0, 16), "avg"), ((6,), "avg"), ((6, 16), "typo")):
            with self.assertRaises(ValueError):
                build_cnn(channels, pooling)

    def test_published_model_matches_independent_100_input_reference(self):
        model, _ = load_selected_model(ROOT)
        expected = json.loads((ROOT / "web/references/runtime-expected.json").read_text(encoding="utf-8"))
        inputs = np.fromfile(ROOT / "web/references/runtime-inputs.bin", dtype="<f4").reshape(-1, 1, 28, 28)
        with torch.no_grad():
            logits = model(torch.from_numpy(inputs.copy()))
            probabilities = logits.softmax(dim=1).numpy()
        np.testing.assert_array_equal(logits.argmax(dim=1).numpy(), expected["top1"])
        np.testing.assert_allclose(probabilities, expected["probabilities"], atol=1e-4, rtol=0)


class MetricTests(unittest.TestCase):
    def test_hand_calculated_asymmetric_matrix_and_absent_prediction(self):
        targets = np.array([0, 0, 0, 1, 1, 2])
        predictions = np.array([0, 0, 1, 1, 0, 0])
        matrix, metrics, report = classification_metrics(targets, predictions, ["A", "B", "C"])
        np.testing.assert_array_equal(matrix, [[2, 1, 0], [1, 1, 0], [1, 0, 0]])
        self.assertEqual(metrics["accuracy_pct"], 50)
        self.assertAlmostEqual(report[0]["precision"], 1 / 2)
        self.assertAlmostEqual(report[0]["recall"], 2 / 3)
        self.assertAlmostEqual(report[0]["f1"], 4 / 7)
        self.assertEqual(report[2]["precision"], 0)
        self.assertEqual(report[2]["support"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], (4 / 7 + 1 / 2) / 3)

    def test_invalid_metric_inputs_are_rejected(self):
        for targets, predictions in (([], []), ([0], [0, 1]), ([0.5], [0]), ([0], [3]), ([-1], [0])):
            with self.subTest(targets=targets, predictions=predictions), self.assertRaises(ValueError):
                classification_metrics(targets, predictions, ["A", "B", "C"])


class SplitTests(unittest.TestCase):
    def test_recorded_split_has_complete_unique_coverage(self):
        train, validation, seed = load_split(ROOT)
        self.assertEqual((len(train), len(validation), seed), (54000, 6000, 42))
        self.assertEqual(np.intersect1d(train, validation).size, 0)

    def test_stratification_is_balanced_and_repeatable(self):
        labels = np.repeat(np.arange(10), 20)
        train, validation = stratified_split(labels, val_per_class=2, seed=42)
        again = stratified_split(labels, val_per_class=2, seed=42)
        np.testing.assert_array_equal(train, again[0])
        np.testing.assert_array_equal(validation, again[1])
        np.testing.assert_array_equal(np.bincount(labels[validation]), np.full(10, 2))
        np.testing.assert_array_equal(np.sort(np.concatenate([train, validation])), np.arange(200))

    def test_corrupt_split_and_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results").mkdir()
            path = root / "results/split_indices.npz"
            train, validation = np.arange(54000), np.arange(54000, 60000)
            for dtype, repeated in ((np.float32, False), (np.int64, True)):
                val = validation.copy()
                if repeated:
                    val[0] = 0
                np.savez(path, train_indices=train.astype(dtype), val_indices=val, seed=42)
                with self.assertRaises(ValueError):
                    load_split(root)
            with self.assertRaisesRegex(ValueError, "recorded experiment"):
                load_split(root, expected_hash="not-the-recorded-file")

    def test_validation_is_separate_and_deterministic(self):
        class Dataset:
            classes = [str(i) for i in range(10)]

            def __init__(self, root, train, download, transform):
                if train is not True or download is not False:
                    raise AssertionError("No test opening or download is expected")
                self.transform = transform

            def __len__(self):
                return 60000

            def __getitem__(self, index):
                image = Image.fromarray(np.arange(784, dtype=np.uint8).reshape(28, 28))
                return self.transform(image), index % 10

        with patch("torchvision.datasets.FashionMNIST", Dataset):
            train, validation, _ = make_loaders(
                ROOT, 42, augmentation=True,
                train_indices=np.array([0, 1, 2]), val_indices=np.array([3, 4]),
            )
            self.assertIsNot(train.dataset.dataset, validation.dataset.dataset)
            first, second = validation.dataset[0][0], validation.dataset[0][0]
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(tuple(first.shape), (1, 28, 28))
            from torchvision.transforms import ToTensor
            self.assertIsInstance(validation.dataset.dataset.transform, ToTensor)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.tensor([[2.0, -1.0], [0.5, 1.0], [-1.0, 0.3]])
        self.targets = torch.tensor([0, 1, 0])
        self.model = nn.Linear(2, 2)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[0.7, -0.2], [-0.1, 0.3]]))
            self.model.bias.zero_()
        self.loader = DataLoader(TensorDataset(self.images, self.targets), batch_size=2)

    def test_evaluation_weights_last_short_batch_by_image_count(self):
        loss, accuracy = evaluate(self.model, self.loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        with torch.no_grad():
            logits = self.model(self.images)
            expected = nn.CrossEntropyLoss()(logits, self.targets).item()
            expected_accuracy = 100 * (logits.argmax(1) == self.targets).sum().item() / 3
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertEqual(accuracy, expected_accuracy)

    def test_evaluation_does_not_change_parameters_or_build_gradients(self):
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        seen = []
        hook = self.model.register_forward_hook(lambda module, inputs, output: seen.append((module.training, output.requires_grad)))
        evaluate(self.model, self.loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        hook.remove()
        self.assertEqual(seen, [(False, False), (False, False)])
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_training_updates_weights_and_covers_last_batch(self):
        before = self.model.weight.detach().clone()
        batch_sizes = []
        hook = self.model.register_forward_hook(lambda module, inputs, output: batch_sizes.append(len(inputs[0])))
        loss, accuracy = train_one_epoch(self.model, self.loader, nn.CrossEntropyLoss(),
                                        torch.optim.SGD(self.model.parameters(), lr=0.1), torch.device("cpu"))
        hook.remove()
        self.assertEqual(batch_sizes, [2, 1])
        self.assertFalse(torch.equal(before, self.model.weight))
        self.assertTrue(np.isfinite(loss) and 0 <= accuracy <= 100)

    def test_empty_datasets_fail_clearly(self):
        empty = DataLoader(TensorDataset(self.images[:0], self.targets[:0]), batch_size=2)
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate(self.model, empty, nn.CrossEntropyLoss(), torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "empty"):
            collect_predictions(self.model, empty.dataset, torch.device("cpu"), 2)


class InferenceAndArtifactTests(unittest.TestCase):
    def test_tensor_prediction_and_invalid_inputs(self):
        model, classes = load_selected_model(ROOT)
        image = np.asarray(Image.open(ROOT / "web/dist/examples/example-0.png"), dtype=np.float32) / 255
        result = predict_tensor(model, torch.from_numpy(image.copy()).unsqueeze(0), classes)
        self.assertEqual(result["predicted_class"], "T-shirt/top")
        self.assertEqual(len(result["top3"]), 3)
        for invalid in (torch.zeros(28, 28), torch.zeros(1, 28, 28, dtype=torch.uint8),
                        torch.full((1, 28, 28), float("nan")), torch.full((1, 28, 28), 2.0)):
            with self.assertRaises(ValueError):
                predict_tensor(model, invalid, classes)

    def test_checkpoint_identity_is_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong.pt"
            path.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_checkpoint(path, expected_hash="recorded-checkpoint")

    def test_atomic_json_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.json"
            save_json(path, {"version": 1})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                save_json(path, {"loss": float("nan")})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_inference_has_no_experiment_or_plotting_imports(self):
        code = "import sys; import fashion_cnn.inference; assert 'matplotlib' not in sys.modules; assert 'fashion_cnn.experiments.search' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ReproductionTests(unittest.TestCase):
    def test_six_stages_receive_an_isolated_workspace(self):
        from contextlib import ExitStack
        from fashion_cnn.experiments import reproduction

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            raw = source / "data/FashionMNIST/raw/train-images-idx3-ubyte"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"source data")
            split = source / "results/split_indices.npz"
            split.parent.mkdir()
            split.write_bytes(b"source split")
            destination = Path(folder) / "new-run"
            names = ["run_experiment", "run_analysis", "run_improvements",
                     "run_stability", "run_evaluation", "explain"]
            with ExitStack() as stack:
                stages = [stack.enter_context(patch.object(reproduction, name)) for name in names]
                reproduction.reproduce(source, destination)
            for stage in stages:
                stage.assert_called_once()
                self.assertEqual(stage.call_args.args[0], destination)
                self.assertTrue(stage.call_args.args[1].is_relative_to(destination))
            self.assertTrue(stages[1].call_args.kwargs["include_final_test"])
            self.assertTrue(stages[4].call_args.kwargs["include_test"])
            self.assertEqual((destination / "data/FashionMNIST/raw" / raw.name).read_bytes(), raw.read_bytes())
            self.assertEqual((destination / "results/split_indices.npz").read_bytes(), split.read_bytes())
            with self.assertRaises(FileExistsError):
                reproduction.reproduce(source, destination)
            with self.assertRaisesRegex(ValueError, "outside"):
                reproduction.reproduce(source, source / "data/recursive-copy")
            self.assertEqual(raw.read_bytes(), b"source data")
            self.assertEqual(split.read_bytes(), b"source split")


class CommandTests(unittest.TestCase):
    def test_help_and_verify_dependencies_are_lightweight(self):
        code = "import sys; from fashion_cnn.cli import make_parser; make_parser().format_help(); from fashion_cnn.artifacts import verify_manifest; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_test_evaluation_is_explicit(self):
        self.assertFalse(make_parser().parse_args(["evaluate"]).test)
        self.assertTrue(make_parser().parse_args(["evaluate", "--test"]).test)

    def test_existing_output_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / "keep.txt"
            sentinel.write_text("original")
            with self.assertRaisesRegex(ValueError, "already exists"):
                fresh_output(ROOT, "unused", folder)
            self.assertEqual(sentinel.read_text(), "original")

    def test_v2_selection_rule(self):
        from fashion_cnn.experiments.search import better
        self.assertTrue(better(91, 0.4, 90, 0.2))
        self.assertTrue(better(91, 0.2, 91, 0.3))
        self.assertFalse(better(90, 0.1, 91, 0.4))
        self.assertFalse(better(91, 0.2, 91, 0.2))


if __name__ == "__main__":
    unittest.main()

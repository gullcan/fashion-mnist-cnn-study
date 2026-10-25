"""One explicit command interface; import only the chosen action's dependencies."""

import argparse
import json
from pathlib import Path


def fresh_output(project_dir, name, requested=None):
    """Generated work belongs in a new directory, not in shipped evidence."""
    path = Path(requested).resolve() if requested else project_dir / "runs" / name
    if path.exists():
        raise ValueError(f"Output already exists: {path}. Choose a new --output directory.")
    return path


def make_parser():
    parser = argparse.ArgumentParser(description="Small Fashion-MNIST CNN study: explicit training, inference and evaluation.")
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="Project root containing data/ and results/.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download data; do not train or evaluate.")
    commands.add_parser("verify", help="Verify the recorded snapshot without ML imports.")
    predict = commands.add_parser("predict", help="Predict one source image with saved weights.")
    predict.add_argument("--index", type=int, default=49804)
    predict.add_argument("--checkpoint", type=Path, help="Optional new-run checkpoint; default is the frozen published model.")
    train = commands.add_parser("train", help="Train the selected 16/32-channel recipe from scratch.")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output", type=Path)
    reproduce = commands.add_parser("reproduce", help="Rerun the complete study: nine training runs, 230 epochs.")
    reproduce.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate the fixed V2 model; validation only by default.")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--test", action="store_true", help="Explicitly reuse the official test benchmark.")
    explain = commands.add_parser("explain", help="Validation activations and already-recorded paired errors.")
    explain.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    root = args.project.resolve()
    try:
        if args.command == "verify":
            from .artifacts import verify_manifest
            verify_manifest(root)
        elif args.command == "download":
            from .data import download_dataset
            download_dataset(root)
        elif args.command == "predict":
            from .inference import predict_sample
            print(json.dumps(predict_sample(root, args.index, args.checkpoint), indent=2))
        elif args.command == "train":
            import torch
            from .data import load_split
            from .experiments.search import train_candidate
            if args.epochs < 1:
                raise ValueError("--epochs must be at least 1")
            train_indices, val_indices, _ = load_split(root)
            config = {"channels": [16, 32], "pooling": "avg", "scheduler": False, "augmentation": True}
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            result, _ = train_candidate(root, fresh_output(root, "train", args.output),
                                        "cnn", "selected V2 recipe; fresh weights", config,
                                        train_indices, val_indices, args.seed, args.epochs,
                                        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print(json.dumps(result, indent=2))
        elif args.command == "reproduce":
            from .experiments.reproduction import reproduce
            reproduce(root, args.output)
        elif args.command == "evaluate":
            from .experiments.evaluation import run_evaluation
            run_evaluation(root, fresh_output(root, "evaluation", args.output), include_test=args.test)
        elif args.command == "explain":
            from .experiments.explanation import explain
            explain(root, fresh_output(root, "explanation", args.output))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")

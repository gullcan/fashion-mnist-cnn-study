# Install, run and reproduce

Run commands from the cloned repository root. Python 3.14 matches the measured environment; the package declares Python >=3.12 because of its pinned dependencies. Other versions/builds are not claimed to reproduce identical numerical results.

## Installation

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Choose one PyTorch build. CPU is enough for prediction and checks:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

The recorded NVIDIA environment used CUDA 13.0 wheels:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

Then install the project. Exact runtime dependency versions live only in `pyproject.toml`; `requirements.txt` is a convenience entry pointing to the same package.

```powershell
python -m pip install -e .
python -m fashion_cnn --help
python -m fashion_cnn verify
python -m fashion_cnn download
```

The download command uses torchvision to obtain the official files. It does not train or evaluate a model. Raw data, the virtual environment and new run outputs are ignored by Git.

## Saved-model prediction

```powershell
python -m fashion_cnn predict --index 49804
```

This loads the frozen seed42 checkpoint on CPU and predicts an official training-source image. Source index 49804 is in the saved validation split. Input indices range from 0 to 59,999. The output includes split membership, real/predicted class and top-three softmax scores; scores are not calibrated correctness probabilities.

## One new training run

```powershell
python -m fashion_cnn train --epochs 30 --seed 42 --output runs/training-01
python -m fashion_cnn predict --index 49804 --checkpoint runs/training-01/best.pt
```

Training starts from new weights using the selected V2 recipe, fixed split, batch64, Adam (learning rate 0.001) and crop/flip augmentation on training data only. Validation selects maximum accuracy, then lower loss. Histories, config, selected weights and curves are saved together. CUDA is used when available. This single run does not repeat the five-candidate search or change the published model.

## Inspect the recorded model

```powershell
python -m fashion_cnn evaluate --output runs/validation-01
python -m fashion_cnn explain --output runs/explanation-01
```

Evaluation checks the fixed checkpoint on all 6,000 validation images and saves predictions, metrics and confusion matrices. Explanation uses validation activations and the already-recorded paired error predictions. Neither command opens the official test set by default. To deliberately repeat the benchmark:

```powershell
python -m fashion_cnn evaluate --test --output runs/benchmark-reuse-01
```

This is reuse of an already-seen official test, not a fresh blind test. Do not use this result to select settings and then call it independent evaluation. Every CLI output destination must be new; choose a new name for another run.

## Full historical campaign

```powershell
python -m fashion_cnn reproduce --output runs/reproduction-01
```

This performs nine training runs totaling 230 epochs: two V1 runs ×10, five V2 candidates ×30, and two additional V2 seeds ×30. It copies downloaded data and the fixed split into the new workspace, then runs V1 analysis, V2 search, stability, frozen evaluation and explanation. It explicitly reuses the official test benchmark as in the original study. The source `results/` is preserved. The destination must not exist and must be outside the source data directory. This is a full campaign, not a quick installation check.

## Tests and browser

```powershell
python -m unittest discover -s tests -v
python -m fashion_cnn verify
node web/tests/test-runtime.mjs
node web/tests/test-preprocess.mjs
```

The tests do not download data, rerun the full training campaign or evaluate the official test set. Node 22 is used by CI; no npm dependencies are needed. See [browser source and local preview](../web/README.md) for the static demo.

## Reproducibility boundaries

`verify` checks SHA256 and byte size for the tracked snapshot listed in `artifact_manifest.json`, excluding the manifest itself. It checks source and recorded artifacts, not ignored local data or future runs. Original report bytes are preserved. When deliberately editing source, regenerate the manifest for a new reviewed snapshot; do not replace original results with invented values.

Deterministic settings and fixed seeds reduce variation within the measured setup. Hardware, libraries and floating-point execution can still change outputs. Checks include saved-checkpoint compatibility, independent browser references and a complete 6,000-image validation pass (91.23% accuracy, 0.9117 macro-F1). The full 230-epoch campaign was not rerun during source maintenance.

For maintainers: review the source diff and confirm that original evidence has not changed, stage intended files, run `python scripts/update_manifest.py`, then stage the manifest. This records the working files selected by Git; it does not authenticate the scientific claims or replace review.

# Running and reproducing the study

## Install

Run commands from the repository root. The recorded environment used Python 3.14.7 on Windows x64. Other platforms have not been validated for identical numerical results.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For the recorded NVIDIA CUDA 13.0 build:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python check_environment.py
```

For CPU-only use, choose the CPU build instead:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`check_environment.py` is specifically an NVIDIA GPU check and deliberately raises an error without CUDA. The other pipeline functions automatically use CPU when CUDA is unavailable; `predict.py` deliberately uses CPU. Neither a new driver nor a separate CUDA Toolkit installation is part of this repository's setup. The recorded PyTorch build uses CUDA 13.0 even though `nvidia-smi` reported a different driver-supported CUDA version.

These are the versions measured in this project. Consult the [official PyTorch installer](https://pytorch.org/get-started/locally/) if a matching wheel is unavailable on your platform. A different version is a new environment and may require adapting pins and the reconstructed-initialization check in `explain_v2.py`.

## Inspect the shipped evidence first

The README, notebook, saved model, reports and images can be inspected without training. Only running a dataset-based prediction needs the raw dataset:

```powershell
python inspect_dataset.py
python predict.py --index 49804
```

`inspect_dataset.py` downloads data once. Torchvision may download test files too; downloading a file is different from using its labels for model selection. The prediction command uses the training-source dataset and does not fit a model. It reports whether the selected source index belongs to the recorded training or validation split.

## Notebook

Jupyter is optional; it is not needed for training or inference scripts:

```powershell
python -m pip install jupyterlab
python -m jupyterlab cnn_project.ipynb
```

The notebook contains saved outputs and can be read on GitHub. Its code cells run sequentially using the saved artifacts; they do not download data, update model weights, launch training or reevaluate the test set. Launch it from the repository root and use this environment's kernel. JupyterLab is an optional reader/editor dependency and was not present in the measured training environment. See the [Jupyter installation documentation](https://docs.jupyter.org/en/latest/install.html).

## Full isolated training reproduction

First download the data. Then:

```powershell
python reproduce.py --output runs/reproduction-01
```

The destination must not exist. The script copies the local Fashion-MNIST data and committed split, then executes the actual pipeline functions in order:

1. `run_experiment`: linear baseline and V1 CNN, 10 epochs each.
2. `run_analysis`: fixed V1 checkpoint validation and official test evaluation.
3. `run_improvements`: five V2 candidates, 30 epochs each, validation-based selection.
4. `run_stability`: two additional seeds, 30 epochs each.
5. `run_evaluation`: selected seed42 V2 checkpoint on the official test.
6. `explain`: validation activation maps and paired errors from saved test predictions.

The total is **230 training epochs** across nine runs, plus evaluations. The original machine completed the five V2 candidates in about 25 minutes of measured training/validation time; two extra seeds take additional time. Timings exclude some startup, file-writing and plotting work. CPU execution can be substantially slower. The copied raw data uses additional disk space; no distributed system or external experiment tracker is used.

The new workspace's `results` folder has the same schema as the shipped `results` folder. Its models are new artifacts: compare them with the original reports rather than claiming exact equality. Cross-device numerical equality is not guaranteed.

The official test has already been used in this study. A rerun is a **benchmark reproduction**, not a new blind holdout. V2 selection uses validation only. The final selected seed is fixed before the stability scores are inspected. Do not choose the highest-scoring test candidate or keep tuning from test results.

## Individual stages and overwritten outputs

The original scripts are retained for inspection and direct use. Their command-line entry points use their own directory as the project root and write to fixed `results/...` destinations. Some training scripts overwrite their own existing outputs; evaluation scripts may read cached test reports when their frozen identity matches. Use `reproduce.py` for a complete rerun in a fresh workspace, rather than sequentially replacing the shipped evidence. `prepare_data.py` is a learning exercise that writes the split and is not needed to use the committed split.

The checkpoint files contain model weights and experiment metadata, not complete optimizer/scheduler states for resuming interrupted training. Reruns start from scratch.

## Artifact identity

`artifact_manifest.json` records SHA256 and size for the packaged files. `verify_artifacts.py` checks them without downloading data or executing training:

```powershell
python verify_artifacts.py
```

The manifest deliberately excludes itself and the verifier. Local extra files, including downloaded data and a virtual environment, are not part of the snapshot. Raw JSON reports keep their original measured values; old absolute paths in provenance metadata describe the original run and are not used by the prediction or reproduction scripts to find files.

## Verification scope

The packaged notebook's code cells were executed in order with the project's Python interpreter. CPU inference on three recorded validation samples agreed with saved CUDA predictions in class and within an absolute softmax-score tolerance of 0.0001; the largest observed difference was about 0.0000301. Repeating the original CUDA batch composition and deterministic settings reproduced these samples' stored probabilities exactly. Small batch/device numerical differences should not be mistaken for retraining or a different checkpoint.

The new reproduction entry point was checked with an isolated control run replacing training stages; the full 230-epoch training campaign was not repeated during packaging. A new Jupyter installation, CPU-only package installation and deployment were not performed during packaging.


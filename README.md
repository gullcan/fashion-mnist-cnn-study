# Fashion-MNIST CNN study

[![Checks](https://github.com/gullcan/fashion-mnist-cnn-study/actions/workflows/checks.yml/badge.svg)](https://github.com/gullcan/fashion-mnist-cnn-study/actions/workflows/checks.yml)

A compact image-classification study comparing a linear baseline and two CNNs, with reproducible training, validation-based model selection and paired error analysis.

**[Live demo](https://gulcan-cnn-kiyafet-laboratuvari.primegull512.chatgpt.site)** · **[Article (Turkish)](https://medium.com/@dgulcan88/bir-bilgisayara-ti%C5%9F%C3%B6rt%C3%BC-nas%C4%B1l-anlat%C4%B1rs%C4%B1n-86493b8b5d4e)**

## Results

All models use the same stratified 54,000/6,000 training/validation split. The official test set contains 10,000 images.

| Model | Parameters | Epoch budget | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| Linear baseline | 7,850 | 10 | 85.52% | 84.55% | 0.8437 |
| V1 CNN | 25,010 | 10 | 88.67% | 88.12% | 0.8807 |
| Selected V2 CNN, seed42 | 57,190 | 30 | 91.23% | **91.08%** | **0.9101** |

The selected model uses 16/32 convolution channels, average pooling, Adam (learning rate 0.001), batch size 64 and training-only random crop/horizontal flip. Five bounded validation experiments compared longer training, a scheduler, max pooling, wider layers and augmentation.

V2 made 892 test errors: 562 V1 errors were corrected and 266 previously correct predictions became incorrect, for a net reduction of 296. Across seeds 42/43/44, validation accuracy averaged 90.27% with a sample standard deviation of 0.844 percentage points on one fixed split. Seed42 was used for configuration selection, so this check is descriptive.

**Evaluation limits:** The official test was reused after V1; it is not a new blind evaluation. Model sizes and training budgets differ. Performance on arbitrary uploaded photographs is unmeasured, and softmax scores are not calibrated correctness probabilities. The architecture is LeNet-inspired, not an exact LeNet-5 reproduction.

![Convolution, ReLU and pooling on validation images](results/explanation_v2/activation_flow.png)

## Quick start

Run from the repository root. Python 3.14 matches the measured environment; CPU is sufficient for inference.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m fashion_cnn verify
python -m fashion_cnn download
python -m fashion_cnn predict --index 49804
```

Train a fresh model or inspect the fixed selected model:

```powershell
python -m fashion_cnn train --epochs 30 --output runs/training-01
python -m fashion_cnn evaluate --output runs/validation-01
python -m fashion_cnn explain --output runs/explanation-01
```

Output directories must be new. Evaluation uses validation data by default; the official test requires explicit `--test`. See [installation and reproduction](docs/REPRODUCING.md) for NVIDIA setup and the full study.

## Implementation

- `fashion_cnn/`: shared model, data, training, inference and classification metrics.
- `fashion_cnn/experiments/`: V1/V2 experiment, selection and evaluation protocols.
- `tests/`: numerical and workflow regression tests.
- `web/`: static browser demo, exported weights and independent runtime checks.
- `results/`: recorded configurations, histories, checkpoints, predictions and figures.

The model factory is shared by training, inference and browser export. Prediction does not import training workflows or plotting. See the [architecture map](docs/CODE_MAP.md), [model card](docs/MODEL_CARD.md) and [browser implementation](web/README.md).

## Checks

```powershell
python -m unittest discover -s tests -v
python -m fashion_cnn verify
node web/tests/test-runtime.mjs
node web/tests/test-preprocess.mjs
```

The 22 Python tests cover split integrity, hand-calculated metrics, sample-weighted loss, parameter-update boundaries, checkpoint compatibility and isolated reproduction. Browser tests compare 100 fixed validation inputs against independent PyTorch references within absolute tolerance 1e-4, and check preprocessing/preview consistency. Runtime parity does not imply 100% classification accuracy. CI runs these checks without downloading the dataset or rerunning the training campaign.

## References and license

- [LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition](https://bottou.org/papers/lecun-98h)
- [Xiao et al. (2017), Fashion-MNIST](https://arxiv.org/abs/1708.07747)

Code is [MIT licensed](LICENSE). Dataset-derived images and examples are attributed in [third-party notices](THIRD_PARTY_NOTICES.md). Raw data, virtual environments and new run outputs are excluded from Git.

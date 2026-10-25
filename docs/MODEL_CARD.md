# Model card: selected Fashion-MNIST V2 CNN

## Purpose and input contract

An educational classifier for ten Fashion-MNIST clothing/accessory categories. Intended uses are studying CNN mechanisms, reproducing a compact experiment, and demonstrating inference on dataset-style images. A [public browser demo](https://gulcan-cnn-kiyafet-laboratuvari.primegull512.chatgpt.site) runs the exported trained weights. Classification performance on arbitrary photos, unseen categories and safety-critical uses was not evaluated.

Input: batch of grayscale `float32` tensors, shape `[N,1,28,28]`, pixel values in `[0,1]`. Benchmark inference uses only `ToTensor()`, without random crop or flip. The optional photo demo adds explicit grayscale, background-polarity and centering heuristics before producing the same tensor shape. Those photo transformations were not used to establish the benchmark accuracy. The training distribution has standardized catalog imagery and background/intensity conventions; simply resizing an arbitrary photograph does not establish compatibility.

Output: ten logits in this order: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. `argmax` selects a class. Softmax is used for inspection; its scores were not calibrated.

## Architecture and training

| Operation | Output per image | Parameters |
|---|---|---:|
| Conv 1 -> 16, 5x5 | 16x24x24 | 416 |
| ReLU, average pool 2x2 | 16x12x12 | 0 |
| Conv 16 -> 32, 5x5 | 32x8x8 | 12,832 |
| ReLU, average pool 2x2 | 32x4x4 | 0 |
| Flatten | 512 | 0 |
| Linear 512 -> 84, ReLU | 84 | 43,092 |
| Linear 84 -> 10 | 10 logits | 850 |
| **Total** | Four learnable layers | **57,190** |

Adam, learning rate 0.001, CrossEntropyLoss, batch64, 30 epochs, seed42. Training only: padded random crop and horizontal flip. No scheduler, dropout, pretrained weights or external data. Selected epoch30 under maximum validation accuracy, then minimum loss on a tie.

Dataset: Fashion-MNIST, official 60,000 training images split into 54,000 training and 6,000 validation, stratified with 600 validation images per class. Official test: 10,000 images, 1,000 per class. Split indices and seed are stored.

## Evaluation

| Split | Accuracy | Macro-F1 | Cross-entropy | Errors |
|---|---:|---:|---:|---:|
| Validation, selected seed42 | 91.2333% | 0.9117333 | 0.2416114 | 526/6,000 |
| Reused official test | 91.0800% | 0.9101046 | 0.2475092 | 892/10,000 |

Metrics include all ten labels; undefined precision/recall/F1 use zero. Confusion rows are true classes, columns predicted classes. Losses are weighted by image count across batches. Online training metrics use changing weights and augmented images; they are not a frozen-model evaluation on the original training set.

Shirt test recall is 71.2%; 132 Shirt images are predicted as T-shirt/top. This is a remaining failure mode, not evidence for tuning from the test.

## Selection and uncertainty

Five candidate configurations were compared on seed42 validation. The final configuration was repeated with seeds43 and44: validation accuracies 89.65% and89.93%, compared with91.23% for seed42. Three-run mean90.2722%, sample standard deviation0.8443 percentage points, on one fixed split.

Seed42 was fixed for the delivered checkpoint before the extra seed results, but it was already the seed used to choose the configuration. The mean is descriptive; it is not an unbiased generalization estimate or a confidence interval. Other candidates were not repeated at the extra seeds. The study does not prove a general augmentation advantage.

The official test was first used for V1 and reused for V2. V2 choices used validation rather than test scores; this is nevertheless not a fresh blind holdout. Different seeds, hardware, versions, data sources and background conventions can change results.

## Identity and provenance

- Checkpoint: `results/improvement_v2/05_augmentation/best.pt`.
- SHA256: `013745dcd66e4d5c462ba64607faed70831ea7e9caa27721afcdc34693b245f2`.
- Split SHA256: `551ae513dfab839035c47ce2832c2570bd193b45ab6e742880b3e1d17c790d19`.
- Test protocol: `results/evaluation_v2/frozen_protocol.json`.
- Configuration: `results/improvement_v2/05_augmentation/config.json`.

V2 before/after filter figures reconstruct the initial weights from the same seed, model factory and PyTorch version; those initial V2 weights were not captured during training. Feature maps are descriptive responses, not causal explanations.

This is inspired by LeNet's local filtering and weight sharing; it is not an exact implementation of the original LeNet-5 architecture or training procedure.


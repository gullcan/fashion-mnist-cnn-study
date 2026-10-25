# Architecture and code map

The reusable package has a small number of plain functions. There is no service layer, plugin registry, training framework or configuration framework. Experiment code orchestrates these functions and preserves the recorded selection rules.

```text
CLI -> experiments -> models / data / engine / metrics / plotting / artifacts
CLI -> inference   -> models / data / artifacts
lessons            -> reusable package (never the reverse)
web export         -> shared Python model -> static browser forward pass
```

`__init__` imports no ML libraries. CLI commands load their dependencies only when selected. Inference loads PyTorch, but no plotting or experiment modules. Plotting is kept out of the numerical model and training loop.

| Responsibility | Implementation |
|---|---|
| Linear model and both CNN sizes | `fashion_cnn/models.py`: `build_baseline`, `build_cnn` |
| Fixed split, class stratification, independent dataset transforms | `fashion_cnn/data.py` |
| Weight updates and sample-weighted evaluation | `fashion_cnn/engine.py` |
| Confusion matrix, precision, recall and macro-F1 | `fashion_cnn/metrics.py` |
| Checkpoint loading and single-image prediction | `fashion_cnn/inference.py` |
| Atomic JSON writes and snapshot hashes | `fashion_cnn/artifacts.py` |
| Figures shared by report workflows | `fashion_cnn/plotting.py` |
| Commands and new-output guard | `fashion_cnn/cli.py` |
| V1 comparison and V2 candidate selection | `fashion_cnn/experiments/v1.py`, `search.py` |
| Additional seeds, frozen evaluation, feature/error inspection | `fashion_cnn/experiments/stability.py`, `evaluation.py`, `explanation.py` |
| Original V1 classification report | `fashion_cnn/analysis.py` |
| Full isolated rerun | `fashion_cnn/experiments/reproduction.py` |
| Numerical and workflow regression checks | `tests/test_core.py` |
| Historical learning exercises | `lessons/` |
| Genuine JavaScript inference and image preparation | `web/dist/inference.mjs`, `preprocess.mjs` |

## Data and model contracts

- Inputs: float32 grayscale `[N,1,28,28]`, scaled to [0,1]. Labels: int64 `[N]`, class IDs 0–9. The network returns logits; cross-entropy receives logits directly.
- The saved split covers each of the 60,000 source indices exactly once, in disjoint 54,000/6,000 arrays. Separate dataset objects prevent training augmentation from leaking into validation.
- Training calls `zero_grad -> forward -> loss -> backward -> step`. Evaluation uses `eval()` and `no_grad()` and never calls an optimizer.
- Batch losses are weighted by image count so that a short last batch does not distort the average.
- Sequential layer order and state-dictionary keys remain compatible with all recorded checkpoints. V1 and V2 use the same model builder with different channel counts.
- Saved-model loading verifies the recorded checkpoint identity. New training weights are distinct artifacts and can be passed explicitly to `predict --checkpoint`.

## Evidence and outputs

`results/` is the original study evidence: `experiment`, `analysis`, `improvement_v2`, `stability_v2`, `evaluation_v2`, `explanation_v2`. These contain the original configs, histories, checkpoint metadata, predictions and reports. Old absolute paths inside provenance JSON describe the original runs; workflows resolve files relative to the supplied project root.

New CLI work goes to fresh `runs/` directories. Full reproduction creates a separate workspace and copies data and the split before running the six study stages. It explicitly repeats the already-used official test benchmark.

Learning exercises are under `lessons/`, guarded by `main()` and write only to `runs/lessons/`. They document development, while the package provides the current shared implementation. Earlier source layout remains available in Git history; no historical commits or metrics were fabricated during the refactor.

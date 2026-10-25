# Code map

| Question | Read |
|---|---|
| What are the original models? | `models.py`: `build_baseline`, `build_cnn` |
| Where do gradients change weights? | `training.py`: `train_one_epoch`; `zero_grad -> forward -> loss -> backward -> step` |
| Where is evaluation separated from learning? | `training.py`: `evaluate`; `model.eval()` and `torch.no_grad()` |
| How is the fixed stratified split made? | `prepare_data.py`; saved `results/split_indices.npz` |
| Where are the 10-epoch comparison experiments? | `run_experiment.py`: `run_experiment`, `make_loaders` |
| How are classification metrics computed? | `analyze_experiment.py`: `classification_metrics`, `collect_predictions`, `analyze_split` |
| How is the final model built? | `improve_cnn.py`: `build_candidate` with the saved candidate config |
| Where is augmentation kept away from validation? | `improve_cnn.py`: `make_loaders`; separate dataset instances |
| What selects a V2 checkpoint? | `improve_cnn.py`: `better`, then `train_candidate` |
| How are extra seeds checked? | `check_v2_stability.py`: `run_stability` |
| What freezes the test model? | `evaluate_v2.py`: `run_evaluation`; identity checks and `frozen_protocol.json` |
| How are intermediate activations shown? | `explain_v2.py`: sequential layer loop inside `explain` |
| How are improvements and regressions paired? | `explain_v2.py`: `compare_errors`; joins on source index |
| How can I predict without training? | `predict.py`: `predict_sample` |
| How can I rerun without replacing recorded results? | `reproduce.py`: fresh output workspace |

The short `convolution_basics.py` and `pooling_basics.py` exercises, and the longer `train.py`, are preserved as learning steps. `train.py` mixes toy demonstrations and one-epoch exercises; it is not the integrated experiment entry point. The notebook and scripts above are the guided reading route. Import the reusable modules rather than importing the top-level learning exercises, which execute on import.

## Evidence locations

- `results/experiment`: V1 configs, histories, selected weights and plots.
- `results/analysis`: V1 validation/test predictions, metrics and frozen protocol.
- `results/improvement_v2`: five candidates and validation-based winner.
- `results/stability_v2`: additional seeds, their histories and aggregate variation.
- `results/evaluation_v2`: selected V2 test results and V1/V2 comparison.
- `results/explanation_v2`: feature maps, filter comparison, paired error transitions and manifests.

Only real project outputs are included. Teacher previews and synthetic flow checks are excluded.


# Deep Learning Assignment 2: CNN Architectures, Training & Ablation Studies

Air University Islamabad. Hifsa Tariq (2600503), Bushra Shahzad (2600508).
Framework: TensorFlow/Keras. Platform: Google Colab, NVIDIA T4 (see `outputs/nvidia_smi.txt`).

## How to reproduce
1. Open `cnn_assignment.ipynb` in Google Colab.
2. Runtime > Change runtime type > T4 GPU.
3. Runtime > Run all (about 30 to 40 minutes). No Google Drive paths are used; everything is written to `outputs/`.
4. To save quota: set `ABL_SEEDS = [42]` and `RUN_LR_SANITY = False` in the configuration cell.

Seed 42 (A0 to A3 also seed 123).

Note: the Colab session was reset after the run, so per-epoch history files, Keras summary files and checkpoints were lost. The CSV/JSON files in `outputs/` were rebuilt from the values printed in the notebook (see `outputs/NOTE.txt`); the notebook's saved outputs are the original record. Data split indices: `outputs/data_indices.json`. Environment: `outputs/environment.txt`, `requirements.txt`.

## Protocol
CIFAR-10, fixed stratified 12,000 train / disjoint 3,000 validation images from the original training set; the 10,000 test images are used once for the preselected final model. 96x96 for the architecture benchmark and A4, 64x64 for A0 to A3 (resized inside tf.data). Adam 3e-4, sparse CE with from_logits=True, mixed_float16 with float32 logits, best-validation-accuracy checkpoint. Four architectures: 5 epochs, batch 32. No failed or OOM runs occurred.

## Repository layout
- `cnn_assignment.ipynb`: full notebook (Parts I to IV)
- `src/models.py`: model definitions
- `outputs/results.csv`, `outputs/results_table.csv`: metrics for every run
- `outputs/*.png`: curves, ablation plot, confusion matrix
- `outputs/lr_sanity.csv`, `outputs/winner.txt`, `outputs/test_result.json`
- `report.pdf`: written report

## Main results (validation accuracy, single run unless noted)
| Model | Val acc |
|---|---|
| AlexNet | 0.576 |
| ResNet-18 | 0.559 |
| VGG16 | 0.539 |
| Inception-v1 | 0.534 |
| A0 / A1 / A2 / A3-P / A3-R (mean of 2 seeds) | 0.358 / 0.432 / 0.438 / 0.644 / 0.611 |
| A4-F / A4-T | 0.837 / 0.868 |

Final model (A4-T, chosen by validation accuracy before test evaluation): test accuracy 0.8566, macro-F1 0.8566.
Single-seed architecture rankings are provisional. See the report for caveats.

Commit hash used for the report: `<paste output of git rev-parse HEAD>`

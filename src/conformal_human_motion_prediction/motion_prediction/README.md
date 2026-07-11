# Motion Prediction Model Training

Training script for the **DCT Pose Transformer**, the human motion-prediction model with
uncertainty quantification used in this repo. The model, loss functions, and the reduced-output
variant live in [`models/dct_pose_transformer_pytorch_attn.py`](../models/dct_pose_transformer_pytorch_attn.py);
the optional RLE (normalising-flow) variant is in
[`models/dct_pose_transformer_rle.py`](../models/dct_pose_transformer_rle.py).

Run everything from the repo root with the venv interpreter (`.venv/bin/python`) and, on GPU,
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (see the top-level [`README.md`](../../../README.md) and
`CLAUDE.md`). Scripts are invoked as modules:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m \
    conformal_human_motion_prediction.motion_prediction.train_motion_prediction_model --stage 1 ...
```

## Overview

The model predicts `PREDICTION_HORIZON_LENGTH = 10` future frames of `N_JOINTS = 13` 3D joints from
`INPUT_HORIZON_LENGTH = 50` input frames, together with a per-joint Gaussian covariance (Cholesky
factor `L`). Training runs in **four stages**, each with its own fresh learning-rate schedule:

1. **Stage 1 — Pose Only.** Train the full transformer to predict poses, no uncertainty.
   - **Trains:** all parameters (transformer blocks, decoders, embeddings).
   - **Loss:** `MAE(pred_poses, target_poses)` (`pose_prediction_loss`).

2. **Stage 2 — Uncertainty Head (frozen backbone).** Learn the covariance head on top of fixed pose
   features. The backbone is frozen by zeroing its gradients in the backward pass.
   - **Trains:** `uncertainty_head` parameters only.
   - **Loss:** `GaussianNLL(pred, target, L)` (`gaussian_nll_from_cholesky`); the pose-loss weight is
     `0`. Add the P2 pinball term with `--lambda_pinball > 0` (see below).

3. **Stage 3 — End-to-End.** Fine-tune the whole model jointly.
   - **Trains:** all parameters.
   - **Loss:** `GaussianNLL + MAE` (`+ λ_pinball · pinball` if enabled).

4. **Stage 4 — End-to-End with Input Uncertainty.** Retrain with the reported *input* uncertainty
   appended to each frame. A fresh model with `input_dim = 156` (`N_JOINTS·3 = 39` pose coords +
   `N_JOINTS·9 = 117` per-joint input covariance) is built, the Stage-3 weights are transferred via
   `merge_params`, and the new input-uncertainty parameters are initialised from scratch.
   - **Trains:** all parameters.
   - **Loss:** `GaussianNLL + MAE` (`+ λ_pinball · pinball`), with optional P4 tail reweighting.
   - **Dataset:** `Human36mMotionDataset3DWithInputUncertainty` (Stages 1–3 use
     `Human36mMotionDataset3D`); the `Augmented` variant is used when `--augment` is set.

### Coverage-calibration options (P2 / P4)

These are **off by default** (the plain pipeline is unchanged). The deployed `final_model`
(`cov_p2p4`) enables both:

- **P2 — set-radius pinball loss** (`--lambda_pinball > 0`, `--set_likelihood τ`): a pinball/quantile
  loss on the deployed spherical set radius `q = sqrt(λ_max(Σ)·χ²₃(τ))`, applied in **Stages 2/3/4**.
  It shapes the covariance directly toward the target coverage `τ` (`set_radius_pinball_loss`).
- **P4 — input-uncertainty tail reweighting** (`--tail_reweight_gamma > 0`, `--tail_reweight_max`):
  up-weights high-input-uncertainty joint-frames in the loss. **Stage 4 only** (needs the input
  covariance block).

## Quick Start

### Full four-stage training (deployed `final_model` recipe)

`--stage 1` runs Stages 1→4 sequentially in one process. This is the canonical recipe kept in sync
with the `Train Motion Prediction Model` entry in [`.vscode/launch.json`](../../../.vscode/launch.json)
and the top-level README:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m \
    conformal_human_motion_prediction.motion_prediction.train_motion_prediction_model \
    --stage 1 --data_path datasets/ --batch_size 256 \
    --d_model 128 --nhead 4 --num_layers 2 \
    --stage1_epochs 80 --stage2_epochs 50 --stage3_epochs 50 --stage4_epochs 50 \
    --learning_rate 0.0001 --use_lr_schedule --lr_schedule_type cosine \
    --lr_warmup_epochs 5 --lr_min_factor 0.1 \
    --weight_decay 0.000001 --max_grad_norm 0.6796845430167515 \
    --augment --max_target_speed 0 --seed 420 \
    --lambda_pinball 1.0 --set_likelihood 0.995 \
    --tail_reweight_gamma 1.0 --tail_reweight_max 5.0 \
    --wandb_project motion-prediction --use_wandb
```

This auto-generates a `run_id` from W&B, trains all four stages, and writes checkpoints + config to
`models/motion_prediction/<run_id>/` (layout below). Turn a finished run into the deployable models
with [`scripts/build_motion_models.py`](../../../scripts/build_motion_models.py):

```bash
python scripts/build_motion_models.py --run_dir models/motion_prediction/<run_id>
# -> models/motion_prediction/final_model/ (full) and final_model_for_ood/ (reduced output)
```

### Resuming / starting from a later stage

`--stage N --resume` loads the latest checkpoint from stage `N-1` and trains stages `N`→4. (With
`--stage 1 --resume` it loads the latest checkpoint found across all stages.) Pass the same
`--run_id` so it finds the run directory and resumes the same W&B run:

```bash
# Continue an interrupted run at Stage 3 (loads the end-of-Stage-2 checkpoint)
python -m conformal_human_motion_prediction.motion_prediction.train_motion_prediction_model \
    --stage 3 --resume --run_id <run_id> --data_path datasets/ ...   # same hyperparameters
```

On resume the existing `dct_pose_transformer_args.json` in the run directory is reloaded, so the
originally saved hyperparameters win over the command line (use `--new_config` to override).

### Quick local test (no W&B, few epochs)

```bash
python -m conformal_human_motion_prediction.motion_prediction.train_motion_prediction_model \
    --run_id test_run --no_wandb --data_path datasets/ \
    --stage1_epochs 2 --stage2_epochs 2 --stage3_epochs 2 --stage4_epochs 2 \
    --batch_size 64 --n_samples 2000
```

### Initialising from transferred weights

`--init_weights_path <pickle>` loads parameters from a pickle (e.g. PyTorch weights transferred to
JAX) via `merge_params` before training, keeping any parameters the source does not provide.

## Command-Line Arguments

Defaults below are the script's argparse defaults; the canonical recipe above overrides several of
them (notably `--batch_size 256`, `--learning_rate 1e-4`, `--use_lr_schedule`, `--augment`, and the
P2/P4 flags).

### Run configuration
- `--stage`: stage to start from, `1`–`4`. Runs this stage through Stage 4 (default: `1`).
- `--run_id`: run identifier (default: W&B run id, or a timestamp when `--no_wandb`).
- `--resume`: resume from the latest checkpoint of stage `stage-1`.
- `--new_config`: ignore an existing saved config and rebuild it from the command line.
- `--init_weights_path`: pickle of initial weights to merge before training (default: none).
- `--n_samples`: cap dataset size for debugging (default: full dataset).

### Model hyperparameters
- `--d_model` (128), `--nhead` (4), `--num_layers` (2)
- `--seq_len` (50, input frames), `--seq_len_output` (10, predicted frames)

### Training hyperparameters
- `--batch_size` (32), `--learning_rate` (1e-3), `--weight_decay` (None), `--max_grad_norm` (None)
- `--use_lr_schedule` (off by default), `--lr_schedule_type` (`cosine` | `exponential`),
  `--lr_warmup_epochs` (5), `--lr_min_factor` (0.01, minimum LR as a fraction of the initial LR)

### Stage epochs
- `--stage1_epochs` (50), `--stage2_epochs` (20), `--stage3_epochs` (30), `--stage4_epochs` (10)

### Data settings
- `--data_path` (`../datasets`), `--seed` (420)
- `--augment`: Z-rotation + scale augmentation on the training set (off by default)
- `--max_target_speed`: too-fast target filter in m/s (default 2.0 = ISO `V_HUMAN`); set `<= 0` to
  disable

### Coverage calibration (P2 / P4)
- `--lambda_pinball` (0.0 = off), `--set_likelihood` (0.995, target coverage τ)
- `--tail_reweight_gamma` (0.0 = off, Stage 4 only), `--tail_reweight_max` (5.0)

### RLE model (optional variant)
- `--use_rle_model`: use `DCTPoseTransformerRLE` (normalising-flow uncertainty head) instead
- `--flow_hidden_dim` (64), `--flow_n_layers` (6, even), `--sigma_init_mm` (20.0)

### Weights & Biases
- `--wandb_project` (`motion-prediction`), `--wandb_entity` (None)
- `--use_wandb` (on by default), `--no_wandb` (disable)

### Optuna hyperparameter search
- `--use_optuna`, `--optuna_n_trials` (100), `--optuna_study_name`, `--optuna_storage`
  (e.g. `sqlite:///optuna.db`), `--optuna_optimize_all_stages`, `--optuna_pruner_warmup` (5),
  `--optuna_pruner_interval` (1). Requires the optional `optuna` dependency (`dev` extra).

## VS Code Debugging

Two ready-made configurations live in [`.vscode/launch.json`](../../../.vscode/launch.json):

1. **Train Motion Prediction Model** — the full four-stage `final_model` recipe above.
2. **Train Motion Prediction with Optuna (All Stages)** — Optuna search over all stages.

Press `F5` and pick one from the dropdown.

## Configuration files

There is no separate hand-maintained config file to edit. On the first run for a `run_id`, the
resolved `TrainingConfig` is written to
`models/motion_prediction/<run_id>/dct_pose_transformer_args.json`; later runs with the same
`run_id` reload it (unless `--new_config`). Each stage export also writes its own
`dct_pose_transformer_args.json` next to the pickle for the eval/OOD pipeline.

## Artifacts & Checkpoints

Checkpoints are written with Orbax every 10 epochs and at the end of each stage. The step number is
the **epoch count within that stage** (not cumulative), so the final checkpoint of a stage is
`checkpoint_<stageN_epochs>`. At each stage end the parameters are also exported as a portable
pickle for the eval/OOD pipeline.

```
models/motion_prediction/<run_id>/
├── dct_pose_transformer_args.json          # resolved run config
└── checkpoints/
    ├── stage_1/
    │   ├── checkpoint_10 … checkpoint_80    # Orbax checkpoints (10-epoch cadence + stage end)
    │   ├── dct_pose_transformer.pickle      # exported weights (portable)
    │   └── dct_pose_transformer_args.json   # exported args for this stage
    ├── stage_2/  …  checkpoint_50
    ├── stage_3/  …  checkpoint_50
    └── stage_4/  …  checkpoint_50
```

(The `checkpoint_N` values shown match the 80/50/50/50 canonical recipe.)

## Logged Metrics (W&B)

Averaged per epoch and logged when `--use_wandb` is set:

- **Train** (`train/…`): `loss`, `nll_loss`, `pinball_loss`, `pose_loss`, `lambda`, plus
  `learning_rate`.
- **Eval** (`eval/…`): `nll_loss`, `pose_loss`, `mpjpe`, `mpjpe_std`, per-horizon
  `mpjpe_time_{80,160,240,320,400}ms`, and `uncertainty_coverage std={1,2,3,4}`.
- **Test** (`test_eval/…`): the same eval metrics computed once on the test split at the end.

## Learning-Rate Scheduling

Each stage gets its **own** schedule, recreated when the stage starts (`--use_lr_schedule`):

1. **Warmup** (`--lr_warmup_epochs`, capped at half the stage): LR ramps `0 → learning_rate`.
2. **Decay** (`cosine` or `exponential`): LR decays `learning_rate → learning_rate · lr_min_factor`
   over the rest of the stage.

So every stage starts a fresh warmup and decays independently — the optimizer is rebuilt between
stages.

## Troubleshooting

- **"No checkpoints found" on resume** — pass the correct `--run_id`, or omit it to use the W&B id.
- **CUDA out of memory** — lower `--batch_size` (e.g. `128` or `64`); on a shared GPU also set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **W&B login required** — run `wandb login`, or pass `--no_wandb`.
- **Config not updating** — an existing run config is reloaded on resume; pass `--new_config` to
  rebuild it from the command line.

## References

- Model + losses: [`models/dct_pose_transformer_pytorch_attn.py`](../models/dct_pose_transformer_pytorch_attn.py)
  (`DCTPoseTransformer`, `pose_prediction_loss`, `gaussian_nll_from_cholesky`,
  `set_radius_pinball_loss`)
- RLE variant: [`models/dct_pose_transformer_rle.py`](../models/dct_pose_transformer_rle.py)
- Dataset dispatch: [`datasets/wrapper.py`](../datasets/wrapper.py) (`dataloader_from_string`)
- Eval metrics: [`utils/eval_utils.py`](../utils/eval_utils.py)
  (`evaluate_pose_prediction_scores_jax`, `evaluate_uncertainty_coverage_jax`)
- Build deployable models: [`scripts/build_motion_models.py`](../../../scripts/build_motion_models.py)
- Settings (joints/horizons/reduced indices): [`h36m_settings.py`](h36m_settings.py)

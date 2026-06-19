# CLAUDE.md

Human **pose-estimation + motion-prediction** pipeline with **sketched-Lanczos OOD scoring**.
`src/`-layout package `conformal_human_motion_prediction`. Full reproduction workflow lives in
[`README.md`](README.md); reference numbers in [`docs/RESULTS.md`](docs/RESULTS.md). This file is
only the non-obvious essentials.

## Environment — read before running anything

- Use the repo venv: `.venv/bin/python`. Install with `pip install -e ".[cuda]"`.
- GPU is an **RTX 5090 (Blackwell, sm_120)**, driver CUDA 12.8. Stack: **jax 0.10, numpy 2, torch cu128**.
  - **torch must be the cu128 build**: `pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`.
    The default PyPI wheel is cu130 and reports `torch.cuda.is_available() == False` on this driver.
  - jax/XLA needs a recent `ptxas`. The venv carries a startup shim
    `<site-packages>/_cuda_ptxas_shim.{py,pth}` that prepends the bundled CUDA-12.9 ptxas to `PATH`.
    It is **not** pip-managed — recreate it if you rebuild the venv, else XLA aborts with
    `PTX version 8.0 does not support target 'sm_120a'`.
  - The GPU is often shared → run with `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid jax's 75 % pre-alloc OOM.
  - **numpy-2 gotcha:** `float(np.array([x]))` raises *"only 0-dimensional arrays…"* — use `.ravel()[0]` / `.item()`.
- **ultralytics** is a custom git-fork dependency (YOLO v26 `Pose26` head with per-keypoint sigma) in
  `pyproject.toml`. Do **not** re-vendor it as a dir/submodule. Verify with `python -c "import ultralytics"`.

## Package layout (`src/conformal_human_motion_prediction/`)

- `datasets/` — H36M / tiger-pose / RGB-D loaders (`dataloader_from_string` dispatch).
- `models/` — JAX model defs (RegressFlow, DCTPoseTransformer; `model_from_string` /
  `pretrained_model_from_string`). **This is the code package, not the artifacts dir.**
- `ood_scoring/` — sketched-Lanczos OOD core + `score_model.py` entrypoint.
- `pose_estimation/`, `motion_prediction/`, `utils/`, `examples/`, `generate_plots/`.

## Artifacts — git-ignored; fetch with `python scripts/download_models.py` (Hugging Face)

- `models/pose_estimation/` — RegressFlow nets, **flat layout** (`jax_resnet50_regressflow*`,
  `jax_resnet18_regressflow*`, `..._3joints*`) + `camera-parameters.json`. `old_models/` = retired variants.
- `models/motion_prediction/` — `final_training_run/` is the canonical (hosted) artifact;
  `final_model/` and `final_model_for_ood/` are **derived** by `python scripts/build_motion_models.py`.
- `models/ood_functions/` — OOD score-function files. `datasets/` — see `datasets/README.md`.

## Running

Run scripts as modules (or via `.vscode/launch.json`):
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m conformal_human_motion_prediction.examples.eval_full_pipeline --enable_ood
```
The pose model and pose OOD are loaded **by direct path** (`--pose_model_path`, `--pose_score_fn_path`),
the same way motion already is (`--motion_model_save_path`, `--motion_score_fn_path`). Exact commands +
expected numbers are in README §3; `final_results/*.sh` regenerate the paper tables.

## OOD scoring (`ood_scoring/score_model.py`)

OOD runs on **reduced-output** models so the GGN/Lanczos stays tractable: pose 3-joint via
`pose_estimation/reduce_regressflow_model.py`; motion `DCTPoseTransformerReducedOutput` (same weights,
output dim 9) via `scripts/build_motion_models.py`. `score_model` does GGN → sketch → Lanczos and writes
the score function to `models/ood_functions/`. Recomputable intermediates cache under `--cache_dir cache/`
(`--load_ggn_vector_product` → `--load_sketch_op` → `--load_eigenpairs` chain); the `base_key` hash keys
**only** those intermediates, not the deliverable. `--sketch_size` and `--lanczos_lm_iter` are the key
paper hyperparameters; lower `--test_batch_size` on GPU-memory issues.

## Notes

- Motion training: `motion_prediction/train_motion_prediction_model.py` (multi-stage; `--augment`).
  `--use_optuna` needs the optional `optuna` (in the `dev` extra).
- Pose nets were trained in PyTorch and weight-transferred to JAX; the deployed pipeline mostly uses the
  YOLO-Pose model, so pose retraining is out of scope here.
- `ros2_packages/` / `ros2_ws/` (real-hardware) are deferred.

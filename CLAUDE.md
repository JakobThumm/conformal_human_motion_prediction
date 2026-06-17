# CLAUDE.md

# Conformal Human Motion Prediction

Human pose-estimation + motion-prediction pipeline with sketched-Lanczos OOD scoring.

## Package layout

`src/`-layout package `conformal_human_motion_prediction`:

- `datasets/` — H36M / tiger-pose / RGB-D loaders. `dataloader_from_string` dispatches by name.
- `models/` — JAX model defs (RegressFlow, DCTPoseTransformer). `model_from_string` /
  `pretrained_model_from_string` dispatch.
- `ood_scoring/` — sketched-Lanczos OOD machinery (`scores/`, `lanczos/`, `sketches/`, `autodiff/`,
  `estimators/`, `losses.py`) and the `score_model.py` entrypoint.
- `pose_estimation/`, `motion_prediction/`, `utils/`, `examples/`, `generate_plots/`.

Checkpoints live in repo-root `models/` (artifacts, git-ignored — see `models/README.md`), data in
repo-root `datasets/` (see `datasets/README.md`). Don't confuse `models/` (artifacts) with the
`conformal_human_motion_prediction.models` code package.

## Setup

`pip install -e .` installs everything, including the **custom ultralytics fork** (YOLO v26
`Pose26` head emitting per-keypoint `sigma_x`/`sigma_y`) via a git dependency in `pyproject.toml`
(`ultralytics @ git+https://github.com/JakobThumm/ultralytics.git@pose-uncertainty-head`). Do **not**
re-add ultralytics as a vendored directory or submodule. Verify: `python -c "import ultralytics"`.

## Data preprocessing

H36M is license-restricted; see `datasets/README.md`. Generate inputs:

```bash
python -m conformal_human_motion_prediction.pose_estimation.preprocess_h36m_bbox_gpu \
    --dataset_dir datasets/H36M/extracted --output_dir datasets/H36M/pre_processed \
    --batch_size 128 --device cuda
```

## Running

Scripts run as modules (or via `.vscode/launch.json` configs):

```bash
python -m conformal_human_motion_prediction.examples.pose_estimation_2D
python -m conformal_human_motion_prediction.examples.pose_estimation_3D --split validation
python -m conformal_human_motion_prediction.examples.eval_full_pipeline --enable_ood
```

## OOD scoring

```bash
python -m conformal_human_motion_prediction.ood_scoring.score_model \
  --ID_dataset H36M --OOD_dataset tiger-pose --data_path datasets/ \
  --model_save_path models/pose_estimation --model RegressFlow \
  --run_name finetuned_h36m_regressflow_pred --subsample_trainset 10000 \
  --lanczos_hm_iter 0 --lanczos_lm_iter 81 --test_batch_size 64 --train_batch_size 64 \
  --serialize_ggn_on_batches --sketch srft --sketch_size 100000
```

- Sketch size and `lm` iterations are the key paper hyperparameters.
- Reduce `--test_batch_size` on GPU-memory issues.

### Caching

Pass `--cache_dir cache/` to save stages; reload to skip work. Dependencies:
`--load_sketch_op` needs `--load_ggn_vector_product`; `--load_eigenpairs` needs both;
`--load_score_functions` is standalone (loads everything, fastest). Cache files are
`*_{ggn,sketch,eigenpairs,score_functions}.cloudpickle`, keyed by dataset/model/run/params.

## Notes

- `ros2_packages/`, `ros2_ws/` (real-hardware) are deferred for a later restructuring pass.
- Model **training** uses `motion_prediction/train_motion_prediction_model.py` (the original
  sketched-Lanczos trainer scripts were removed; only `ood_scoring/losses.py` survives, for the
  GGN/Hessian loss definitions).

# Conformal Human Motion Prediction

A human **pose-estimation + motion-prediction** pipeline with **sketched-Lanczos OOD scoring**
and conformal uncertainty quantification. Given camera images, it estimates 2D/3D human poses
(RegressFlow), predicts future motion (DCTPoseTransformer), and flags out-of-distribution inputs
using a low-memory sketched-Lanczos Laplace approximation.

## Repository structure

```
src/conformal_human_motion_prediction/
├── datasets/          # H36M / tiger-pose / RGB-D dataset loaders (shared)
├── models/            # JAX model definitions: RegressFlow, DCTPoseTransformer (shared)
├── ood_scoring/       # sketched-Lanczos OOD machinery + score_model.py entrypoint
│   ├── score_model.py, losses.py
│   └── scores/ lanczos/ sketches/ autodiff/ estimators/
├── pose_estimation/   # 2D/3D pose inference, preprocessing, triangulation
├── motion_prediction/ # motion model training, inference, covariance evaluation
├── utils/             # transforms, GPU utils, visualization, pose_metrics
├── examples/          # runnable demo / debug / evaluation scripts
└── generate_plots/    # paper result/plot generation

models/      # checkpoint artifacts (git-ignored) — see models/README.md
datasets/    # dataset artifacts (git-ignored) — see datasets/README.md
scripts/     # download_models.py
final_results/  tests/  docs/  docker/  ros2_packages/  ros2_ws/
```

## Setup

```bash
# Editable install. This pulls the custom ultralytics fork (YOLO v26 Pose26 head with
# per-keypoint uncertainty sigma_x/sigma_y) directly from git — no manual clone needed.
python3 -m pip install -e .

# verify the fork is active
python3 -c "import ultralytics; print(ultralytics.__version__)"
```

### GPU (CUDA) setup

The base install pulls a CPU build of JAX. For GPU, install the `cuda` extra and a CUDA-matched
PyTorch build:

```bash
# CUDA 12 JAX plugin (+ bundled CUDA/cuDNN wheels)
python3 -m pip install -e ".[cuda]"

# PyTorch must match your NVIDIA driver. Plain `pip install torch` may pull a cu130 wheel that
# needs a CUDA-13 driver; install from the index matching your driver instead, e.g. cu128:
python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

On newer GPUs (RTX 50xx / Blackwell, sm_120) XLA needs a recent `ptxas` (CUDA ≥ 12.9, PTX ISA
≥ 8.7); an older system `/usr/bin/ptxas` triggers `PTX version 8.0 does not support target
'sm_120a'`. The `nvidia-cuda-nvcc-cu12` wheel ships a new-enough ptxas, but it must win over the
system one on `PATH`. This repo's venv carries a startup shim
(`<site-packages>/_cuda_ptxas_shim.{py,pth}`) that prepends the bundled ptxas automatically. The
shim is not managed by pip — if you rebuild the venv, recreate it (or just
`export PATH=<site-packages>/nvidia/cuda_nvcc/bin:$PATH` before running).

Then fetch data and checkpoints:

- **Datasets**: see [`datasets/README.md`](datasets/README.md) (H36M is license-restricted; the
  preprocessing commands generate the pose/motion inputs).
- **Models**: `python scripts/download_models.py` — see [`models/README.md`](models/README.md)
  (configure `MODEL_MANIFEST` with the checkpoint URLs first).

## Running the pipeline

Most entrypoints are runnable as modules or via the VSCode launch configs in `.vscode/launch.json`.

**2D pose estimation:**
```bash
python -m conformal_human_motion_prediction.examples.pose_estimation_2D
```

**2D pose on preprocessed H36M:**
```bash
python -m conformal_human_motion_prediction.examples.evaluate_preprocessed_h36m \
    --preprocessed_dir datasets/H36M/pre_processed \
    --checkpoint models/pose_estimation/H36M/RegressFlow/seed_420 \
    --split validation --num_samples 100 --visualize --save_dir results/preprocessed_eval_vis
```

**3D pose estimation:**
```bash
python -m conformal_human_motion_prediction.examples.pose_estimation_3D
```

**ID vs. OOD pose prediction:**
```bash
python -m conformal_human_motion_prediction.examples.id_vs_ood_pose_prediction
```

**Full pipeline evaluation:**
```bash
python -m conformal_human_motion_prediction.examples.eval_full_pipeline --enable_ood
```

## OOD scoring (sketched Lanczos)

Build the OOD score function with `score_model`:

```bash
python -m conformal_human_motion_prediction.ood_scoring.score_model \
  --ID_dataset H36M --OOD_dataset tiger-pose --data_path datasets/ \
  --model_save_path models/pose_estimation --model RegressFlow \
  --run_name finetuned_h36m_regressflow_pred \
  --subsample_trainset 10000 --lanczos_hm_iter 0 --lanczos_lm_iter 81 \
  --test_batch_size 64 --train_batch_size 64 --serialize_ggn_on_batches \
  --sketch srft --sketch_size 100000
```

- Sketch size and number of `lm` iterations are the key hyperparameters (discussed in the paper).
- Reduce `--test_batch_size` if you hit GPU memory limits in `score_fun(X)`.

### Caching intermediate computations

The Lanczos pipeline is expensive (20+ min JIT compile for large models). Cache stages with
`--cache_dir cache/` and reload them to speed up tuning:

| Flag | Skips |
|------|-------|
| `--load_ggn_vector_product` | ~20 min GGN JIT compilation |
| `--load_sketch_op` (needs GGN) | sketch creation |
| `--load_eigenpairs` (needs GGN + sketch) | Lanczos iterations |
| `--load_score_functions` | entire score-building phase (fastest) |

Cache files use the `.cloudpickle` extension, one per stage
(`*_ggn`, `*_sketch`, `*_eigenpairs`, `*_score_functions`), keyed by dataset/model/run/params.

## Known issues

**CuDNN version mismatch** (`Loaded runtime CuDNN library: 9.x but source was compiled with 9.y`):
install a matching CuDNN locally and prepend it to `LD_LIBRARY_PATH`.

**JAX/Flax/Optax version mismatch** (e.g. `'EvalTrace' object has no attribute 'level'`):
`pip install --upgrade jax jaxlib flax optax`. Tested working: jax/jaxlib 0.7.0, flax 0.8.4, optax 0.2.2.

## ROS2 / real-hardware

`ros2_packages/` and `ros2_ws/` run the pipeline on real hardware. They depend on two external
packages (`realsense_rgbd_streamer`, `uq_msgs`, hosted under `github.com/JakobThumm`) that were
previously git submodules; re-add them when restructuring the ROS2 stack.

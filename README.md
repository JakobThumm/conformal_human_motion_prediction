# Conformal Human Motion Prediction

A human **pose-estimation + motion-prediction** pipeline with **sketched-Lanczos OOD scoring**
and conformal uncertainty quantification. Given camera images, it estimates 2D/3D human poses
(RegressFlow / YOLO-Pose), predicts future motion (DCTPoseTransformer), and flags out-of-distribution
inputs using a low-memory sketched-Lanczos Laplace approximation.

## ROS2 / real-hardware

To run the conformal human motion prediction (`chmp`) in ROS2, we provide the following repositories:

- [`chmp_workspace`](https://github.com/JakobThumm/chmp_workspace) — docker + scripts that assemble
  a runnable ROS2 workspace, dev-mounting this repo as an editable install and optionally pulling
  `realsense_rgbd_streamer` for RealSense hardware.
- [`chmp_inference`](https://github.com/JakobThumm/chmp_inference) — the ROS2 node package that
  runs this pipeline (depends on `uq_msgs` for custom messages).

## Repository structure

```
src/conformal_human_motion_prediction/
├── datasets/          # H36M / tiger-pose / RGB-D dataset loaders (shared)
├── models/            # JAX model definitions: RegressFlow, DCTPoseTransformer (shared)
├── ood_scoring/       # sketched-Lanczos OOD machinery + score_model.py entrypoint
├── pose_estimation/   # 2D/3D pose inference, preprocessing, triangulation, model reduction
├── motion_prediction/ # motion model training, inference, covariance evaluation
├── utils/             # transforms, GPU utils, visualization, pose_metrics
├── examples/          # runnable demo / debug / evaluation scripts
└── generate_plots/    # paper result/plot generation

models/      # checkpoint artifacts (git-ignored) — fetch with scripts/download_models.py
datasets/    # dataset artifacts (git-ignored) — see datasets/README.md
scripts/     # download_models.py, upload_models.py, build_motion_models.py
docs/RESULTS.md   # detailed reference numbers for every evaluation
```

## Setup

```bash
# Editable install. Pulls the custom ultralytics fork (YOLO v26 Pose26 head with
# per-keypoint uncertainty sigma_x/sigma_y) directly from git — no manual clone needed.
python3 -m pip install -e .
python3 -c "import ultralytics; print(ultralytics.__version__)"   # verify the fork is active
```

### GPU (CUDA) setup

The base install pulls a CPU build of JAX. For GPU, install the `cuda` extra and a CUDA-matched
PyTorch build:

```bash
python3 -m pip install -e ".[cuda]"        # CUDA 12 JAX plugin (+ bundled CUDA/cuDNN wheels)

# PyTorch must match your NVIDIA driver. Plain `pip install torch` may pull a cu130 wheel that
# needs a CUDA-13 driver; install from the index matching your driver instead, e.g. cu128:
python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

On newer GPUs (RTX 50xx / Blackwell, sm_120) XLA needs a recent `ptxas` (CUDA ≥ 12.9); an older
system `/usr/bin/ptxas` triggers `PTX version 8.0 does not support target 'sm_120a'`. The
`nvidia-cuda-nvcc-cu12` wheel ships a new-enough ptxas, but it must win over the system one on
`PATH`. This repo's venv carries a startup shim (`<site-packages>/_cuda_ptxas_shim.{py,pth}`) that
prepends the bundled ptxas automatically; if you rebuild the venv, recreate it (or
`export PATH=<site-packages>/nvidia/cuda_nvcc/bin:$PATH`).

Tested working stack: jax/jaxlib 0.10.1, flax 0.12, optax 0.2.8, numpy 2.x, torch 2.x+cu128.

---

# Reproducing the results

The pipeline has three ingredients: **datasets**, **models**, and the **OOD score functions** built
from them. The quickest path to results is: download the models (§2.1), fetch + preprocess H36M
(§1), then run the evaluation scripts (§3). Training (§2.2) and OOD-function building (§2.3) are only
needed if you want to regenerate the artifacts from scratch.

## 1. Datasets

### 1.1 Get H36M

Human3.6M is license-restricted, so you must download it yourself. Follow
[anibali/h36m-fetch](https://github.com/anibali/h36m-fetch) to download and extract the data, then
place (or symlink) the extracted tree at `datasets/H36M/extracted/`, e.g.:

```
datasets/H36M/extracted/<subject>/Videos/<action>.<camera>.mp4
datasets/H36M/extracted/<subject>/Poses_D2_Positions/<action>.<camera>.cdf
datasets/H36M/extracted/<subject>/Poses_D3_Positions/<action>.cdf
```

See [`datasets/README.md`](datasets/README.md) for the expected layout. `tiger-pose` (the pose OOD
set) and the RGB-D lab recordings are fetched separately (also documented there).

### 1.2 Preprocess the image / pose data (for pose estimation + pose OOD)

The RegressFlow pose models and the pose OOD scoring run on cropped, bbox-normalised frames.
Generate them once with the GPU preprocessor (YOLO-detect → crop → resize → store as uint8):

```bash
python -m conformal_human_motion_prediction.pose_estimation.preprocess_h36m_bbox_gpu \
    --dataset_dir datasets/H36M/extracted \
    --output_dir  datasets/H36M/pre_processed \
    --batch_size 128 --device cuda
```

This writes `datasets/H36M/pre_processed/<subject>/PreprocessedImages/*.npy` (+ `PreprocessedPoses/*.npz`).
If a run is interrupted it can leave empty `.npy` files; the loaders skip empty/corrupt files with a
warning, and you can regenerate just the gaps by re-running the preprocessor (it overwrites).

### 1.3 Motion data

The **standard** motion model reads 3D pose sequences **directly** from
`datasets/H36M/extracted/<subject>/Poses_D3_Positions/` — **no preprocessing step is required** for
motion training/evaluation.

The **uncertainty-input** variant (motion conditioned on *predicted* 3D poses + covariances, as in
the full pipeline) needs a one-off preprocessing pass that runs pose estimation over H36M and stores
the predicted poses with covariance:

```bash
python -m conformal_human_motion_prediction.motion_prediction.preprocess_uncertainty_input_dataset \
    --data_path datasets/ --pose_model_path models/pose_estimation/jax_resnet50_regressflow
```

## 2. Models

### 2.1 Download the models

Checkpoints are hosted on the Hugging Face Hub. Fetch and assemble them with:

```bash
python scripts/download_models.py          # all groups, then builds the deployable motion models
# options: --only pose_estimation | --only motion_prediction | --no-build | --repo_id <ns/name>
```

This downloads `models/pose_estimation/` (RegressFlow nets + `camera-parameters.json`) and
`models/motion_prediction/final_training_run/` (per-stage checkpoints), then derives
`final_model/` and `final_model_for_ood/` locally via `scripts/build_motion_models.py` (so the
reduced-output OOD config stays in sync with the code). See [`models/README.md`](models/README.md).

### 2.2 Train the models

**Pose estimation — not retrained here.** The RegressFlow pose nets were originally trained with
Marian's PyTorch code and the weights transferred to JAX. In the deployed pipeline we primarily use
the **YOLO-Pose** model for pose estimation, so retraining RegressFlow is out of scope for this repo
— just download the checkpoints (§2.1).

**Motion prediction.** The DCTPoseTransformer is trained in stages (pose-only → uncertainty head →
end-to-end → final), with optional Z-rotation/scale augmentation (`--augment`):

```bash
python -m conformal_human_motion_prediction.motion_prediction.train_motion_prediction_model \
    --stage 1 --data_path datasets/ --batch_size 256 \
    --d_model 128 --nhead 4 --num_layers 2 \
    --stage1_epochs 15 --stage2_epochs 10 --stage3_epochs 10 --stage4_epochs 40 \
    --learning_rate 0.0001 --use_lr_schedule --lr_schedule_type cosine \
    --lr_warmup_epochs 3 --lr_min_factor 0.1 \
    --weight_decay 0.000001 --max_grad_norm 0.6796845430167515 \
    --augment --wandb_project motion-prediction --use_wandb
```

Checkpoints land in `models/motion_prediction/<run_id>/checkpoints/stage_*/`. Turn a finished run
into the deployable models with:

```bash
python scripts/build_motion_models.py --run_dir models/motion_prediction/<run_id>
# -> writes models/motion_prediction/final_model/ (full) and final_model_for_ood/ (reduced output)
```

### 2.3 Build the OOD scoring functions [Optional]

The OOD scoring functions are already downloaded from huggingface, but you can rebuilt them like described here.

**Reduced models — what they are.** Sketched-Lanczos OOD scoring builds a Laplace/GGN approximation
whose cost scales with the model's **output dimension**. To keep it tractable, OOD is computed on
**reduced-output** versions of the models that predict only the safety-critical joints:

- **Pose** (`reduce_regressflow_model.py`): slices the RegressFlow output head from 17 joints → 3
  (nose + both wrists, indices `[0, 9, 10]`), i.e. output dim 34 → 6. This is a real weight
  transformation:
  ```bash
  python -m conformal_human_motion_prediction.pose_estimation.reduce_regressflow_model \
      --run_name jax_resnet18_regressflow --output_run_name jax_resnet18_regressflow_3joints --seed 420
  ```
- **Motion** (`DCTPoseTransformerReducedOutput`): **same weights**, output sliced to timestep
  `REDUCED_TIMESTEP=4` and joints `REDUCED_JOINT_INDICES=[0,5,6]` (head + both hands), i.e. output dim
  1560 → 9. `scripts/build_motion_models.py` produces this as `final_model_for_ood/` (identical
  `*.pickle` weights + an `args.json` selecting `DCTPoseTransformerReducedOutput`).

**Build the score function.** `score_model` computes the GGN, sketches it, runs Lanczos, and saves a
single (small) score-function file. Pose OOD (ID = H36M, OOD = tiger-pose):

```bash
python -m conformal_human_motion_prediction.ood_scoring.score_model \
    --ID_dataset H36M --OOD_datasets tiger-pose --data_path datasets/ \
    --model_save_path models/pose_estimation --model RegressFlowResNet18_3Joints \
    --run_name jax_resnet18_regressflow_3joints --output_dim 6 \
    --subsample_trainset 10000 --lanczos_hm_iter 0 --lanczos_lm_iter 81 \
    --test_batch_size 64 --train_batch_size 64 --serialize_ggn_on_batches \
    --sketch srft --sketch_size 100000 --cache_dir cache/ \
    --score_fn_output_path models/ood_functions/pose_score_fn.cloudpickle
```

Motion OOD (ID = augmented reduced H36M motion, OOD = the reduced OOD motion set):

```bash
python -m conformal_human_motion_prediction.ood_scoring.score_model \
    --ID_dataset Human36mMotionReducedOutputDataset3DAugmented \
    --OOD_datasets Human36mMotionReducedOutputOODDataset3D --data_path datasets/ \
    --model_save_path models/motion_prediction/final_model_for_ood --model DCTPoseTransformerReducedOutput \
    --run_name dct_pose_transformer --output_dim 9 \
    --subsample_trainset 10000 --lanczos_hm_iter 0 --lanczos_lm_iter 800 \
    --test_batch_size 128 --train_batch_size 128 --serialize_ggn_on_batches \
    --sketch srft --sketch_size 10000 --cache_dir cache/ \
    --score_fn_output_path models/ood_functions/motion_score_fn.cloudpickle
```

Notes:
- `--sketch_size` and `--lanczos_lm_iter` are the key paper hyperparameters.
- The expensive intermediates (GGN / sketch / eigenpairs) are cached under `--cache_dir cache/`;
  reload them with `--load_ggn_vector_product` / `--load_sketch_op` / `--load_eigenpairs` (each needs
  the previous), or `--load_score_functions` to skip the whole build. Reduce `--test_batch_size` if
  you hit GPU-memory limits.

## 3. Results

### 3.1 Paper results (`final_results/`)

The headline tables in the paper are produced by the three scripts in `final_results/`. Each runs
the relevant evaluation over the H36M **test** split, then builds a LaTeX table from the saved CSVs
(via `generate_plots/`). Run them once the models and OOD score functions are in place (§2):

```bash
bash final_results/motion_prediction_no_uncertainty_no_ood_results.sh    # motion benchmark
bash final_results/motion_prediction_conformal_prediction_set_results.sh # conformal prediction sets
bash final_results/full_pipeline_ood_handling_evaluation.sh              # full pipeline (sweeps N_req)
```

**Motion prediction test results** (`motion_prediction_no_uncertainty_no_ood_results.sh`) — MPJPE
[mm] from ground-truth pose inputs; reports the stage-1 and final models:

| Method | 80 ms | 160 ms | 320 ms | 400 ms |
|--------|-------|--------|--------|--------|
| Ours (stage 1) | **8.7** | **16.7** | **41.1** | **54.5** |
| Ours (final)   | 18.4 | 28.1 | 53.1 | 67.2 |

**Conformal prediction set test results** (`motion_prediction_conformal_prediction_set_results.sh`) —
predicted pose inputs with input uncertainty:

| Method | ↑ Coverage (%) | ↓ Volume (m³) |
|--------|----------------|----------------|
| ISO 13855:2010 | 97.93 | 0.191 |
| Conformal prediction sets (ours) | **98.25** | **0.017** |

**Full pipeline evaluation** (`full_pipeline_ood_handling_evaluation.sh`) — sweeps `N_req`, the number
of correct poses required before triggering motion prediction (the script also runs `N_req=5`):

| N_req | ↓ ℋ invalid (%) | ↑ Motion valid (%) | ↓ MPJPE (mm) |
|-------|------------------|---------------------|---------------|
| 3 (ours) | **9.45** | **85.48** | 53.56 |
| 10 | 11.72 | 82.10 | 53.13 |
| 50 | 14.75 | 75.29 | **52.22** |

### 3.2 Per-stage sanity checks

Faster per-component runs (validation split, small subsets) to confirm each stage works. All
evaluation scripts load the pose model and OOD score function **by path** (`--pose_model_path`,
`--pose_score_fn_path`, `--motion_model_save_path`, `--motion_score_fn_path`). Full per-joint /
per-time breakdowns are in [`docs/RESULTS.md`](docs/RESULTS.md):

| Stage | Command | Expected |
|-------|---------|----------|
| **2D pose (JAX)** | `python -m conformal_human_motion_prediction.examples.pose_estimation_2D` | MPJPE ≈ **7.7 px** |
| **2D pose (YOLO-Pose)** | `python -m conformal_human_motion_prediction.examples.pose_estimation_2D_yolo` | MPJPE ≈ **13.7 px** |
| **3D pose full eval** | `python -m conformal_human_motion_prediction.examples.pose_estimation_3D_full_eval --max_sequences 10 --split validation` | MPJPE ≈ **32–36 mm** |
| **Motion prediction** | `python -m conformal_human_motion_prediction.examples.motion_prediction --split validation` | MPJPE ≈ **48 mm**; 1σ cov ≈ 57 % |
| **Full pipeline** | `python -m conformal_human_motion_prediction.examples.eval_full_pipeline --action Directions --max_sequences 1 --enable_ood` | 3D pose **19.68 mm**, motion ≈ **34.6 mm** |
| **Covariance calibration** | `python -m conformal_human_motion_prediction.motion_prediction.evaluate_covariance --results_file results/motion_prediction/motion_prediction_results_validation.cloudpickle --config h36m` | tuned 1σ cov ≈ **85 %** |

The full pipeline takes the pose model + OOD and the motion model + OOD explicitly, e.g.:

```bash
python -m conformal_human_motion_prediction.examples.eval_full_pipeline \
    --pose_model_path     models/pose_estimation/jax_resnet50_regressflow \
    --pose_score_fn_path  models/ood_functions/pose_score_fn.cloudpickle \
    --motion_model_save_path models/motion_prediction/final_model/dct_pose_transformer.pickle \
    --motion_score_fn_path   models/ood_functions/motion_score_fn.cloudpickle \
    --split validation --action Directions --max_sequences 1 --enable_ood
```

### 3.3 Most relevant example scripts

Quick, single-purpose scripts for testing/debugging individual pipeline stages (most also have a
VSCode launch config in `.vscode/launch.json`):

| Script | Purpose |
|--------|---------|
| `examples/pose_estimation_2D.py` / `_2D_yolo.py` | 2D pose on H36M (RegressFlow / YOLO-Pose) |
| `examples/pose_estimation_3D.py` | 3D pose via stereo triangulation |
| `examples/pose_estimation_2d_with_ood.py` | 2D pose + per-frame OOD flagging |
| `examples/id_vs_ood_pose_prediction.py` | pose OOD separability (H36M vs tiger-pose) |
| `examples/motion_prediction.py` | motion-prediction evaluation + coverage |
| `examples/id_vs_ood_motion_prediction.py` | motion OOD separability |
| `examples/eval_full_pipeline.py` | end-to-end H36M: pose → motion → OOD → coverage |
| `examples/eval_full_pipeline_rgbd_yolo.py` | end-to-end on RGB-D lab recordings (YOLO) |
| `examples/debug_pose_visualization.py` / `debug_3d_pose_visualization.py` | render predicted vs GT poses |
| `examples/debug_rgbd_2d_visualization.py` / `debug_rgbd_3d_visualization.py` | inspect RGB-D frames + predictions |

---

## Known issues

- **CuDNN version mismatch** (`Loaded runtime CuDNN library: 9.x but source was compiled with 9.y`):
  install a matching CuDNN and prepend it to `LD_LIBRARY_PATH`.
- **`PTX version ... does not support target 'sm_120a'`**: the ptxas shim wasn't on `PATH` — see the
  GPU setup section.
- **Out-of-memory from JAX** on a shared GPU: JAX pre-allocates 75 % of VRAM by default; set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or `XLA_PYTHON_CLIENT_MEM_FRACTION`).
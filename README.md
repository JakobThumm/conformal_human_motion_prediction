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
end-to-end → final), with optional Z-rotation/scale augmentation (`--augment`). This is the deployed
`final_model` (cov_p2p4) recipe: the set-radius pinball loss (`--lambda_pinball`, `--set_likelihood`)
and Stage-4 input-uncertainty tail reweighting (`--tail_reweight_gamma`, `--tail_reweight_max`)
self-calibrate the predicted covariance toward the target coverage.

```bash
python -m conformal_human_motion_prediction.motion_prediction.train_motion_prediction_model \
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

### 2.4 Calibrate the conformal prediction sets

The conditional-conformal calibrator
(`models/motion_prediction/conformal_calibration/conformal_calibrator.npz`) turns the model's raw
covariances into the deployed prediction sets. It is fit on the **validation** split in two steps.

**Step 1 — evaluate motion prediction on the validation split** (with OOD, so the per-sample score
is measured) to produce the cloudpickle the calibrator reads. VSCode launch config
*"Motion Prediction Evaluation"*:

```jsonc
{
  "name": "Motion Prediction Evaluation",
  "type": "debugpy",
  "request": "launch",
  "program": "src/conformal_human_motion_prediction/examples/motion_prediction.py",
  "console": "integratedTerminal",
  "args": [
    "--data_path", "datasets/",
    "--dataset_name", "Human36mMotionDataset3DWithInputUncertainty",
    "--split", "validation",
    "--model_save_path", "models/motion_prediction/final_model/dct_pose_transformer.pickle",
    "--enable_ood",
    "--motion_score_fn_path", "models/ood_functions/dct_pose_transformer_score_fn.cloudpickle"
  ]
}
```

Equivalently on the CLI:

```bash
python -m conformal_human_motion_prediction.examples.motion_prediction \
    --data_path datasets/ --dataset_name Human36mMotionDataset3DWithInputUncertainty \
    --split validation \
    --model_save_path models/motion_prediction/final_model/dct_pose_transformer.pickle \
    --enable_ood --motion_score_fn_path models/ood_functions/dct_pose_transformer_score_fn.cloudpickle
```

**Step 2 — fit the calibrator** (tune the target confidence here). VSCode launch config
*"Conformal Calibration (tune confidence)"*:

```jsonc
{
  "name": "Conformal Calibration (tune confidence)",
  "type": "debugpy",
  "request": "launch",
  "program": "src/conformal_human_motion_prediction/motion_prediction/conformal_calibration.py",
  "console": "integratedTerminal",
  "env": {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "JAX_PLATFORMS": "cpu"  // pure numpy/scipy; keep off the shared GPU
  },
  "args": [
    // Tuning phase: split validation 50/50 (calib/test) so coverage is reported honestly.
    // Final step: drop --calib_frac and pass --calib_file <S11> --test_file <S5> instead.
    "--results_file", "results/motion_prediction/motion_prediction_results_validation.cloudpickle",
    "--calib_frac", "0.5",
    // >>> Play with the target confidence here (overrides SET_LIKELIHOOD). <<<
    "--likelihood", "0.9999"
    // Other knobs you can add: --n_bins 8  --tail_edges 0.3 0.5 0.75  --n_min 200
    //                         --base raw|affine  --no-monotone  --seed 0
  ]
}
```

Equivalently on the CLI:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu \
  python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
    --results_file results/motion_prediction/motion_prediction_results_validation.cloudpickle \
    --calib_frac 0.5 --likelihood 0.9999
```

## 3. Results

### 3.1 Paper results (`final_results/`)

The headline tables in the paper are produced by the scripts in `final_results/`. Each runs the
relevant evaluation over the H36M **test** split, then builds a LaTeX table from the saved CSVs
(via `generate_plots/`). Run them once the models and OOD score functions are in place (§2), and
after fitting the calibrator (§2.4):

```bash
bash final_results/motion_prediction_no_uncertainty_no_ood_results.sh     # motion benchmark (MPJPE)
bash final_results/motion_prediction_conformal_prediction_set_results.sh  # conformal prediction sets (coverage/volume)
bash final_results/robot_shield_safety_results.sh                         # robot-shield certification (c_safe, PFH_D, PL)
bash final_results/full_pipeline_ood_handling_evaluation.sh               # full pipeline (sweeps N_req)
bash final_results/create_full_conformal_prediction_results_table.sh      # combine conformal + shield -> final table
```

The conformal prediction-set and robot-shield scripts each cover all three methods (ISO 13855,
ours with OOD inputs, ours with OOD filtering); the last script fuses their CSVs into the combined
`tab:all_conformal_results`. Each script also writes its paper-ready `.tex` under `results/final/`.

**Motion prediction test results** (`motion_prediction_no_uncertainty_no_ood_results.sh`) — MPJPE
[mm] from ground-truth pose inputs; reports the stage-1 and final models:

| Method | 80 ms | 160 ms | 320 ms | 400 ms |
|--------|-------|--------|--------|--------|
| Ours (stage 1) | **8.7** | **16.7** | **41.1** | **54.5** |
| Ours (final)   | 18.4 | 28.1 | 53.1 | 67.2 |

**Conformal prediction set test results** (`motion_prediction_conformal_prediction_set_results.sh`) —
predicted pose inputs with input uncertainty; a single test-set evaluation yields all three methods.
Volume is the 5/50/95 percentiles of the per-sphere volume (robust to the heavy OOD tail):

| Method | ↑ Coverage (%) | ↓ Vol 5% (m³) | ↓ 50% | ↓ 95% |
|--------|----------------|---------------|-------|-------|
| ISO 13855 (no OOD filter) | 99.9193 | 0.017 | 0.687 | 3.252 |
| Ours (no OOD filter)      | 99.9785 | 0.015 | 0.091 | 0.664 |
| Ours (OOD filtered)       | **99.9835** | **0.015** | **0.088** | **0.638** |

**Robot-shield certification** (`robot_shield_safety_results.sh`) — three shield runs at
$N \approx \num{2e13}$ simulated HRC test cycles ($t_\text{cycle}=4$ ms); `c_safe` is the verified
rate, `c_safe ∧ contact` the dangerous failures, PFH_D the one-sided Clopper-Pearson (99.99%) upper
bound. See §3.4 for the workflow:

| Method | ↑ c_safe (%) | ↓ c_safe ∧ contact | ↓ PFH_D (1/h) | PL |
|--------|--------------|--------------------|---------------|-----|
| ISO 13855 (no OOD filter) | 98.91 | 12,206,306 | 1.49e-1 | none |
| Ours (no OOD filter)      | 99.21 | 2 | 6.27e-7 | PL d |
| Ours (OOD filtered)       | **99.23** | **0** | **4.14e-7** | **PL d** |

**Full pipeline evaluation** (`full_pipeline_ood_handling_evaluation.sh`) — sweeps `N_req`, the number
of correct poses required before triggering motion prediction:

| N_req | ↓ ℋ invalid (%) | ↑ Motion valid (%) | ↓ MPJPE (mm) |
|-------|------------------|---------------------|---------------|
| 3 (ours) | **12.63** | **74.74** | 55.15 |
| 5 | 13.52 | 73.62 | 55.03 |
| 10 | 14.05 | 72.67 | 54.74 |
| 50 | 16.57 | 68.70 | **54.14** |

> Our OOD pipeline reduces the rate of invalid pose buffers
> $\sum_{i=K_I - N_\text{req}+1}^{K_I} v_i < N_\text{req}$ by 23.8 % while only increasing the average
> MPJPE by 1.9 %.

**Combined final results table** (`create_full_conformal_prediction_results_table.sh`,
`tab:all_conformal_results`) — coverage reported as miss-rate ($1-p_\text{cov}$) and nines of
reliability ($-\log_{10}$ miss-rate):

| Method | ↓ Miss-rate | ↑ 9s of rel. | Vol 5% | 50% | 95% | ↑ c_safe (%) | ↓ ∧ contact | ↓ PFH_D (1/h) | PL |
|--------|-------------|--------------|--------|-----|-----|--------------|-------------|---------------|-----|
| ISO 13855            | 8.1e-4 | 3.09 | 0.017 | 0.687 | 3.252 | 98.91 | 1.2e7 | 1.49e-1 | none |
| Ours with OOD inputs | 2.1e-4 | 3.67 | 0.015 | 0.091 | 0.664 | 99.21 | 2 | 6.27e-7 | PL d |
| Ours OOD filtered    | **1.6e-4** | **3.80** | **0.015** | **0.088** | **0.638** | **99.23** | **0** | **4.14e-7** | **PL d** |

The paper-ready LaTeX for every table above is written under `results/final/` (e.g.
`results/final/all_conformal_results.tex`).

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

### 3.4 Robot-shield safety evaluation workflow

End-to-end evaluation of the motion model as a SARA-style safety shield: drop a robot into the
recorded human scenes and measure how often the shield verifies a trajectory as safe while the
ground truth has an (unsafe) contact. The verified-but-unsafe rate is bounded
(Clopper-Pearson) and converted to an ISO 13849-1 PFH_D / Performance Level.

The full chain, one step per artifact:

1. **Train the motion model** — `motion_prediction.train_motion_prediction_model` (use the
   **P2+P4** setup, launch config *"Train Motion Prediction Model P2+P4 (cov_p2p4, final)"*: P2
   set-radius pinball loss + P4 tail reweighting self-calibrate the covariance). Training writes
   only checkpoints under `models/motion_prediction/<run>/checkpoints/stage_*/`. Promote a run to
   the deployed model with `python scripts/build_motion_models.py --run_dir
   models/motion_prediction/<run>` (derives `final_model/` and `final_model_for_ood/`).
2. **Predict the full eval set** — `examples.motion_prediction --split validation` saves
   `results/motion_prediction/motion_prediction_results_<split>.cloudpickle` (predictions, targets,
   raw covariances, input uncertainty, OOD scores). **This step is required:** training does *not*
   emit predictions, only checkpoints. (Build the motion OOD score function first — see §2.3 *Build
   the OOD scoring functions* — so `--enable_ood` can fill in the scores.)
3. **Calibrate the conformal sets** — `motion_prediction.conformal_calibration --results_file <the
   cloudpickle> --likelihood <target coverage>` fits the conditional-conformal calibrator
   (`results/motion_prediction/conformal_calibration/conformal_calibrator.npz`).
4. **Eval data for the shield** — the shield consumes the **raw** predictions/covariances from
   step 2 and applies the calibrator itself, so when calibration and evaluation share a split this
   reuses step 2's cloudpickle (no re-prediction). For an honest train/test split, predict the test
   split separately (`--split test`) and point the shield at that cloudpickle.
5. **Simulate the robot shield** — `examples.simulate_robot_shield --backend gpu
   --conformal_calibrator <npz> --results_file <cloudpickle> --num_robot_poses 1000000
   --pose_radius 10 --results_csv results/final/robot_shield/shield_results.csv` appends one summary
   row (headline rates + PFH_D / PL per confidence) per run. Turn the CSV into a LaTeX table with
   `generate_plots.generate_robot_shield_results`.

VSCode launch config *"Simulate Robot Shield"* (settings: OOD threshold `1.5E-5`, set likelihood
`0.9999`):

```jsonc
{
  "name": "Simulate Robot Shield",
  "type": "debugpy",
  "request": "launch",
  "program": "src/conformal_human_motion_prediction/examples/simulate_robot_shield.py",
  "console": "integratedTerminal",
  "env": {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
    // Do NOT set JAX_PLATFORMS here: the script auto-keeps the GPU for "--backend gpu" and
    // forces JAX onto the CPU otherwise. (Add "JAX_PLATFORMS": "cpu" only to pin everything,
    // incl. the kernel, to the CPU backend.)
  },
  "args": [
    // 4 ms robot CSV (--robot_csv) and --robot_stride 25 are the defaults.
    "--results_file", "results/motion_prediction/motion_prediction_results_test.cloudpickle",
    "--num_robot_poses", "1000000",   // production: ~1E6 (needed for PL_C); the GPU path culls ~85% for free.
    "--pose_radius", "10.0",       // production: 10.0 (far poses then cull at level 1 on the CPU).
    "--pose_z_offset", "0.2",
    "--seed", "0",
    "--mask_ood",
    "--backend", "gpu",  // Run verification of heavy intersection poses on GPU.
    "--gpu_dtype", "float32",
    "--gpu_a_chunk", "512",
    "--save_failures", "results/motion_prediction/shield_failures.npy", "--max_failures", "200"
  ]
}
```

For the paper table set a target cycle count with `--n_test_cycles 2e13` (the script derives
`--num_robot_poses`) and select the human occupancy model with `--human_set {conformal,sara}`
(`sara` = the ISO 13855 constant-velocity reachable set); the three rows differ by `--human_set`
and `--mask_ood`.

Steps 2–5 (plus the LaTeX table for all three methods) are scripted in
[`final_results/robot_shield_safety_results.sh`](final_results/robot_shield_safety_results.sh)
(prerequisites: step 1 + the motion OOD score function). The shield runs the fine-phase
intersection checks on the GPU (`--backend gpu`); levels 1–3 of the bounding-sphere culling stay on
the CPU. `--backend cpu --num_workers N` is the reference path. Verify the two agree with
`--parity N`.

---

## Known issues

- **CuDNN version mismatch** (`Loaded runtime CuDNN library: 9.x but source was compiled with 9.y`):
  install a matching CuDNN and prepend it to `LD_LIBRARY_PATH`.
- **`PTX version ... does not support target 'sm_120a'`**: the ptxas shim wasn't on `PATH` — see the
  GPU setup section.
- **Out-of-memory from JAX** on a shared GPU: JAX pre-allocates 75 % of VRAM by default; set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or `XLA_PYTHON_CLIENT_MEM_FRACTION`).
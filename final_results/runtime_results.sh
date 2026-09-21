#!/bin/bash
# Table: per-stage runtime of the full pipeline (tab:runtime).
#
# Replays the per-frame work of examples/eval_full_pipeline on H36M two-camera test data and times
# the six stages separately (2D pose estimation, pose OOD, triangulation, motion prediction, motion
# OOD, prediction-set computation). Every stage is synchronised before its timestamp is taken, and
# the first $WARMUP measurements per stage are discarded -- JIT compilation alone costs seconds
# (10-15 s for the two jax models), so un-warmed numbers are meaningless.
#
# The CSV also carries a `measured_wall` row: the independent end-to-end wall clock per step. It
# should sit a few ms above the stage sum (the untimed buffer bookkeeping); a large gap means the
# per-stage synchronisation is wrong.
set -e

POSE_MODEL="models/pose_estimation/jax_resnet50_regressflow"
POSE_SCORE_FN="models/ood_functions/jax_resnet18_regressflow_3joints_score_fn.cloudpickle"
MOTION_MODEL="models/motion_prediction/final_model/dct_pose_transformer.pickle"
MOTION_SCORE_FN="models/ood_functions/dct_pose_transformer_randproj_score_fn.cloudpickle"
OUTPUT_DIR="results/final/runtime"
N_FRAMES="${N_FRAMES:-500}"
WARMUP="${WARMUP:-50}"

# Run this on an otherwise idle GPU -- a shared GPU inflates every stage and widens the std.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m conformal_human_motion_prediction.examples.benchmark_pipeline_runtime \
    --pose_model_path "$POSE_MODEL" \
    --pose_score_fn_path "$POSE_SCORE_FN" \
    --motion_model_save_path "$MOTION_MODEL" \
    --motion_score_fn_path "$MOTION_SCORE_FN" \
    --split "test" \
    --n_frames "$N_FRAMES" \
    --warmup "$WARMUP" \
    --output_dir "$OUTPUT_DIR"

python -m conformal_human_motion_prediction.generate_plots.generate_runtime_results \
    --results_dir "$OUTPUT_DIR"

#!/bin/bash
# Table: Full pipeline evaluation results on H36M (H invalid %, Motion valid %, MPJPE mm)
# for several values of N_req (number of correct poses required before motion prediction).
set -e

POSE_MODEL="models/pose_estimation/jax_resnet50_regressflow"
POSE_SCORE_FN="models/ood_functions/pose_score_fn.cloudpickle"
MOTION_MODEL="models/motion_prediction/final_model/dct_pose_transformer.pickle"
MOTION_SCORE_FN="models/ood_functions/motion_score_fn.cloudpickle"

for N in 3 5 10 50; do
    python -m conformal_human_motion_prediction.examples.eval_full_pipeline \
        --pose_model_path "$POSE_MODEL" \
        --pose_score_fn_path "$POSE_SCORE_FN" \
        --motion_model_save_path "$MOTION_MODEL" \
        --motion_score_fn_path "$MOTION_SCORE_FN" \
        --split "test" \
        --enable_ood \
        --n_correct_poses_required "$N" \
        --output_dir "results/final/full_pipeline/n_correct_poses_required_${N}/"
done

# Build the LaTeX table from the saved CSVs.
python -m conformal_human_motion_prediction.generate_plots.generate_full_pipeline_results

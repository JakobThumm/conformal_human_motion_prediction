#!/bin/bash
# Robot-shield safety evaluation (ISO 13849-1 PFH_D / Performance Level) for the final motion model.
#
# Runs workflow steps 2)-5) end to end and emits a LaTeX results table:
#   2) predict the full eval set            -> motion_prediction_results_<split>.cloudpickle
#   3) calibrate the conformal sets         -> conformal_calibrator.npz   (per set likelihood)
#   4) (eval data for the shield)           -> reuses step 2's cloudpickle (see note below)
#   5) simulate the robot shield            -> one CSV row per set likelihood
#   +  build the LaTeX table from that CSV
#
# Prerequisites (NOT run here):
#   * step 1) a trained motion model exported to models/motion_prediction/final_model/
#       (derive it from a training run with: python scripts/build_motion_models.py
#        --run_dir models/motion_prediction/cov_p2p4)
#   * the motion OOD score function at models/ood_functions/dct_pose_transformer_score_fn.cloudpickle
#       (build it with the ood_scoring.score_model command in the README).
#
# Note on "predict twice": the shield consumes the RAW predictions/covariances; the conformal
# calibrator is applied inside the shield, not baked into the predictions. So when calibration and
# evaluation use the SAME split (the default validation 50/50 here), step 2 produces the eval data
# once and step 4 just reuses it. For an honest train/test split, predict the test split separately
# (--split test) and point --results_file / --test_file at it; see the commented block below.
set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false   # share the GPU politely (jax pre-alloc off)

SPLIT="${SPLIT:-validation}"
MODEL="${MODEL:-models/motion_prediction/final_model/dct_pose_transformer.pickle}"
SCORE_FN="${SCORE_FN:-models/ood_functions/dct_pose_transformer_score_fn.cloudpickle}"
RESULTS="results/motion_prediction/motion_prediction_results_${SPLIT}.cloudpickle"
CALIB="results/motion_prediction/conformal_calibration/conformal_calibrator.npz"
CSV="results/final/robot_shield/shield_results.csv"

# Set likelihoods to sweep (the conformal target coverage). Override with: LIKELIHOODS="0.999 0.9999"
LIKELIHOODS="${LIKELIHOODS:-0.999 0.9995 0.9999}"
NUM_POSES="${NUM_POSES:-1000000}"   # ~1e6 needed for PL_C; lower for a quick check
POSE_RADIUS="${POSE_RADIUS:-10.0}"

mkdir -p "$(dirname "$CSV")"
rm -f "$CSV"                          # fresh sweep -> one row per likelihood below

# 2) Predict the full eval set (RAW predictions + covariances + input uncertainty + OOD scores).
python -m conformal_human_motion_prediction.examples.motion_prediction \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3DWithInputUncertainty \
  --split "$SPLIT" \
  --model_save_path "$MODEL" \
  --enable_ood \
  --motion_score_fn_path "$SCORE_FN" \
  --output_dir results/motion_prediction

for L in $LIKELIHOODS; do
  echo "==================== set likelihood $L ===================="

  # 3) Calibrate the conditional-conformal sets at this target coverage.
  python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
    --results_file "$RESULTS" \
    --calib_frac 0.5 \
    --likelihood "$L"

  # 4) Eval data for the shield == step 2's cloudpickle (same split). For an honest split, instead:
  #    python -m ...examples.motion_prediction --split test ... --output_dir results/motion_prediction
  #    and set RESULTS to the test cloudpickle.

  # 5) Simulate the robot shield and append one summary row to the CSV.
  python -m conformal_human_motion_prediction.examples.simulate_robot_shield \
    --results_file "$RESULTS" \
    --conformal_calibrator "$CALIB" \
    --backend gpu --gpu_dtype float32 --gpu_a_chunk 512 \
    --num_robot_poses "$NUM_POSES" \
    --pose_radius "$POSE_RADIUS" --pose_z_offset 0.2 \
    --robot_stride 25 --mask_ood --seed 0 \
    --results_csv "$CSV"
done

# 6) Build the LaTeX table (one row per set likelihood, best PFH_D bolded).
python -m conformal_human_motion_prediction.generate_plots.generate_robot_shield_results \
  --csv "$CSV" \
  --output results/final/robot_shield/robot_shield_safety.tex \
  --confidence 0.9999

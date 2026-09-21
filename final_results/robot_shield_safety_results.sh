#!/bin/bash
# Table 2/3: Robot-shield safety / certification simulation (ISO 13849-1 PFH_D / Performance Level)
# on H36M, for the methods of the final results table. Each shield run targets the same number
# of simulated HRC test cycles (N ~= N_TEST_CYCLES); the number of random robot poses is derived per
# run so N stays fixed even as OOD/too-fast filtering changes the eligible human-sample count.
#
#   * ISO 13855 without OOD filtered : --human_set sara  --no-mask_ood  (constant-velocity, v=2 m/s)
#   * Ours without OOD filtered      : --human_set conformal --no-mask_ood
#   * Ours with OOD filtered         : --human_set conformal --mask_ood
#   * Ours (alpha_max) w/o + with OOD filtered : the same two runs driven by $CALIB_MAX, the
#       single-threshold ABLATION of our conditional calibration (one global alpha_max from the
#       per-sample max normalized error; r = alpha_max * sqrt(lambda_max(C))). The calibrator .npz
#       carries its own mode, so only the --conformal_calibrator path changes; the run tags itself
#       set_kind=max_conformal in the CSV and lands on its own table row.
#   * Ours (no calib.) w/o + with OOD filtered : the same two runs driven by $CALIB_UNC, the
#       NO-CALIBRATION ablation -- calibration is skipped entirely and the predicted covariance is
#       trusted with the fixed training-time factor alpha = sqrt(chi2_3(1-eps)) = 4.5943 at
#       1-eps = 0.9999 (the 3-dof Mahalanobis ellipsoid collapsed to a sphere via lambda_max).
#       Tags itself set_kind=uncalibrated in the CSV. No conformal guarantee.
#
# RUN ORDER IS A PRIORITY ORDER. The runs are sequenced so that an interrupted sweep still leaves
# the rows the paper needs most, and the CSV is appended to (not wiped) by default -- the table
# generators take the LAST row per method, so re-running a single method just supersedes it:
#   1) Ours OOD filtered            4) Ours (no calib.) OOD filtered
#   2) ISO 13855                    5) Ours with OOD inputs
#   3) Ours (alpha_max) OOD filtered
# The two remaining cells (alpha_max / no-calib. with OOD *inputs*) are not planned for the paper
# and only run with RUN_OPTIONAL=1.
#
# Knobs:
#   FRESH=1         wipe $CSV first (default 0: append, so a partial sweep survives)
#   RUN_OPTIONAL=1  also run the two OOD-inputs ablation cells (default 0)
#   SKIP_CALIB=1    reuse the existing calibrator .npz files instead of refitting (default 0;
#                   refitting is deterministic at --calib_frac 0.5, so resuming is safe either way)
#   PAPER_METHODS   method keys the LaTeX table shows (default: the five above)
#
# Steps: predict eval+test set -> write all three set definitions -> shield runs (one CSV row
# each) -> build the standalone shield LaTeX table.
#
# Prerequisites (NOT run here): a trained motion model at $MODEL and the motion OOD score function
# at $SCORE_FN (see README).
#
# NOTE: $SCORE_FN and OOD_THRESHOLD in motion_prediction/h36m_settings.py are head-specific and
# must match. The default score function is the random-projection head (scores in metres, ID mean
# ~0); the legacy fixed-joints head scores in millimetres (ID mean ~9e4). Re-tune OOD_THRESHOLD
# after changing heads, otherwise OOD masking silently never fires.
set -e

# export XLA_PYTHON_CLIENT_PREALLOCATE=false   # share the GPU politely (jax pre-alloc off)

MODEL="${MODEL:-models/motion_prediction/final_model/dct_pose_transformer.pickle}"
SCORE_FN="${SCORE_FN:-models/ood_functions/dct_pose_transformer_randproj_score_fn.cloudpickle}"
CALIB="models/motion_prediction/conformal_calibration/conformal_calibrator.npz"
CALIB_MAX="models/motion_prediction/conformal_calibration/conformal_calibrator_max.npz"
CALIB_UNC="models/motion_prediction/conformal_calibration/conformal_calibrator_uncalibrated.npz"
CSV="results/final/robot_shield/shield_results.csv"
LIKELIHOOD="${LIKELIHOOD:-0.9999}"
N_TEST_CYCLES="${N_TEST_CYCLES:-2e13}"   # target simulated HRC test cycles per method
POSE_RADIUS="${POSE_RADIUS:-10.0}"
FRESH="${FRESH:-0}"                   # 1 = wipe the CSV; 0 = append (last row per method wins)
RUN_OPTIONAL="${RUN_OPTIONAL:-0}"     # 1 = also run the two OOD-inputs ablation cells
SKIP_CALIB="${SKIP_CALIB:-0}"         # 1 = reuse the existing calibrator .npz files
PAPER_METHODS="${PAPER_METHODS:-iso_no_ood ours_no_ood ours_uncal_ood ours_max_ood ours_ood}"

mkdir -p "$(dirname "$CSV")"
if [ "$FRESH" != "0" ]; then
  rm -f "$CSV"
fi

# 2) Predict the eval and test set (RAW predictions + covariances + input uncertainty + OOD scores).
# python -m conformal_human_motion_prediction.examples.motion_prediction \
#   --data_path datasets/ \
#   --dataset_name Human36mMotionDataset3DWithInputUncertainty \
#   --split validation \
#   --model_save_path "$MODEL" \
#   --enable_ood \
#   --motion_score_fn_path "$SCORE_FN" \
#   --output_dir results/motion_prediction
# 
# python -m conformal_human_motion_prediction.examples.motion_prediction \
#   --data_path datasets/ \
#   --dataset_name Human36mMotionDataset3DWithInputUncertainty \
#   --split test \
#   --model_save_path "$MODEL" \
#   --enable_ood \
#   --motion_score_fn_path "$SCORE_FN" \
#   --output_dir results/motion_prediction

RESULTS_EVAL="results/motion_prediction/motion_prediction_results_validation.cloudpickle"
RESULTS_TEST="results/motion_prediction/motion_prediction_results_test.cloudpickle"

if [ "$SKIP_CALIB" = "0" ]; then
  # 3) Write all three set definitions at the target coverage, from the same validation run and the
  #    same calib/test sample split: the conditional q_hat grid, the max-score ablation, and the
  #    no-calibration ablation (which consumes no calibration data -- alpha = sqrt(chi2_3(level)) --
  #    but goes through the same writer so it deploys as a drop-in calibrator .npz).
  python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
    --results_file "$RESULTS_EVAL" \
    --calib_frac 0.5 \
    --likelihood "$LIKELIHOOD" \
    --method conditional \
    --calibrator_path "$CALIB"

  python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
    --results_file "$RESULTS_EVAL" \
    --calib_frac 0.5 \
    --likelihood "$LIKELIHOOD" \
    --method max \
    --calibrator_path "$CALIB_MAX" \
    --output_dir results/motion_prediction/conformal_calibration_max

  python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
    --results_file "$RESULTS_EVAL" \
    --calib_frac 0.5 \
    --likelihood "$LIKELIHOOD" \
    --method uncalibrated \
    --calibrator_path "$CALIB_UNC" \
    --output_dir results/motion_prediction/conformal_calibration_uncalibrated
fi

# 4)+5) Shield runs on the test set -> one CSV row per method, in priority order.
#   common knobs: derive poses from N_TEST_CYCLES, GPU backend, decorrelated cycles.
#   --gpu_a_chunk trades padding waste against kernel launches; 512 suits the full test set, but
#   re-tune it if a run's active-motion count is much smaller (the shield prints the distribution).
COMMON="--results_file $RESULTS_TEST \
  --backend gpu --gpu_dtype float32 --gpu_a_chunk 512 \
  --n_test_cycles $N_TEST_CYCLES --pose_radius $POSE_RADIUS --pose_z_offset 0.2 \
  --robot_stride 25 --seed 0 --results_csv $CSV"

run_shield () {                       # $1 = banner, rest = per-method flags
  local banner="$1"; shift
  echo "==================== $banner ===================="
  python -m conformal_human_motion_prediction.examples.simulate_robot_shield $COMMON "$@"
}

run_shield "1/5  Ours OOD filtered" \
  --conformal_calibrator "$CALIB" --human_set conformal --mask_ood

run_shield "2/5  ISO 13855 (SARA, v=2 m/s)" \
  --human_set sara --no-mask_ood

run_shield "3/5  Ours (alpha_max ablation) OOD filtered" \
  --conformal_calibrator "$CALIB_MAX" --human_set conformal --mask_ood

run_shield "4/5  Ours (no calibration ablation) OOD filtered" \
  --conformal_calibrator "$CALIB_UNC" --human_set conformal --mask_ood

run_shield "5/5  Ours with OOD inputs" \
  --conformal_calibrator "$CALIB" --human_set conformal --no-mask_ood

if [ "$RUN_OPTIONAL" != "0" ]; then   # not planned for the paper
  run_shield "opt  Ours (alpha_max ablation) with OOD inputs" \
    --conformal_calibrator "$CALIB_MAX" --human_set conformal --no-mask_ood

  run_shield "opt  Ours (no calibration ablation) with OOD inputs" \
    --conformal_calibrator "$CALIB_UNC" --human_set conformal --no-mask_ood
fi

# 6) Build the standalone robot-shield LaTeX table (one row per method that is in the CSV).
python -m conformal_human_motion_prediction.generate_plots.generate_robot_shield_results \
  --csv "$CSV" \
  --output results/final/robot_shield/robot_shield_safety.tex \
  --methods "$PAPER_METHODS" \
  --confidence 0.99999

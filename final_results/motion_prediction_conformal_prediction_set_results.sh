#!/bin/bash
# Table 1/3: Conformal prediction set test results on H36M (Coverage % / Volume m^3), for the
# methods of the final results table:
#   * ISO 13855 without OOD filtered  -> SARA constant-velocity reachable set (coverage_stats_sara.csv)
#   * Ours without OOD filtered       -> conditional-conformal set, all test samples
#   * Ours with OOD filtered          -> conditional-conformal set, in-distribution samples only
#   * Ours (alpha_max) w/o and with OOD filtered -> the single-threshold ABLATION of the two rows
#       above: one global alpha_max = conformal quantile of the per-sample max normalized error
#       A_max = max_{j,k} ||d_k^j|| / sqrt(lambda_max(C_k^j)), applied as r = alpha_max *
#       sqrt(lambda_max(C_k^j)). All 130 joint-timestep sets then hold simultaneously at 1-eps,
#       at the cost of all conditioning on input uncertainty / joint / horizon.
#   * Ours (no calib.) w/o and with OOD filtered -> the NO-CALIBRATION ablation: skip calibration
#       entirely and trust the predicted covariance with the fixed training-time factor
#       alpha = sqrt(chi2_3(1-eps)) (4.5943 at 1-eps = 0.9999), i.e. the 3-dof Mahalanobis
#       ellipsoid at 1-eps collapsed to a sphere via lambda_max. No conformal guarantee.
#
# All rows come from a SINGLE evaluation run of examples.motion_prediction on the test set (with
# --enable_ood so the per-sample OOD score is measured); the evaluation part derives every
# coverage/volume row from that one run. The calibrators are prerequisites: each is (re)fitted here
# from a validation run unless it already exists.
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
OUTPUT_DIR="results/final/conformal_prediction_sets"
LIKELIHOOD="${LIKELIHOOD:-0.9999}"

mkdir -p "$OUTPUT_DIR"

# 1) Calibrate on the validation split unless all calibrators already exist. The conditional
#    calibrator and both ablations are fitted from the same validation run (and the same
#    calib/test sample split), so they differ only in how the radius is formed. The
#    no-calibration ablation uses no calibration data at all (alpha = sqrt(chi2_3(level))); it
#    still goes through the same writer so it deploys as a drop-in calibrator .npz.
VALIDATION_RESULTS="$OUTPUT_DIR/motion_prediction_results_validation.cloudpickle"
if { [ ! -f "$CALIB" ] || [ ! -f "$CALIB_MAX" ] || [ ! -f "$CALIB_UNC" ]; } && [ ! -f "$VALIDATION_RESULTS" ]; then
  echo "==================== predicting the validation split ===================="
  python -m conformal_human_motion_prediction.examples.motion_prediction \
    --data_path datasets/ \
    --dataset_name Human36mMotionDataset3DWithInputUncertainty \
    --split validation \
    --model_save_path "$MODEL" \
    --output_dir "$OUTPUT_DIR"
fi
if [ ! -f "$CALIB" ]; then
  echo "==================== fitting conformal calibrator (conditional) ===================="
  python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
    --results_file "$VALIDATION_RESULTS" \
    --calib_frac 0.5 \
    --likelihood "$LIKELIHOOD" \
    --method conditional \
    --calibrator_path "$CALIB"
fi
if [ ! -f "$CALIB_MAX" ]; then
  echo "==================== fitting conformal calibrator (max-score ablation) ===================="
  python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
    --results_file "$VALIDATION_RESULTS" \
    --calib_frac 0.5 \
    --likelihood "$LIKELIHOOD" \
    --method max \
    --calibrator_path "$CALIB_MAX" \
    --output_dir results/motion_prediction/conformal_calibration_max
fi
if [ ! -f "$CALIB_UNC" ]; then
  echo "==================== writing calibrator (no-calibration ablation) ===================="
  python -m conformal_human_motion_prediction.motion_prediction.conformal_calibration \
    --results_file "$VALIDATION_RESULTS" \
    --calib_frac 0.5 \
    --likelihood "$LIKELIHOOD" \
    --method uncalibrated \
    --calibrator_path "$CALIB_UNC" \
    --output_dir results/motion_prediction/conformal_calibration_uncalibrated
fi

# 2) The single evaluation run: predict the test set with OOD measurement. The eval part writes the
#    per-method coverage CSVs into $OUTPUT_DIR:
#      coverage_stats_sara.csv
#      coverage_stats_conformal_prediction_sets.csv
#      coverage_stats_conformal_prediction_sets_ood_filtered.csv
#      coverage_stats_conformal_prediction_sets_max.csv
#      coverage_stats_conformal_prediction_sets_max_ood_filtered.csv
#      coverage_stats_conformal_prediction_sets_uncalibrated.csv
#      coverage_stats_conformal_prediction_sets_uncalibrated_ood_filtered.csv
python -m conformal_human_motion_prediction.examples.motion_prediction \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3DWithInputUncertainty \
  --split test \
  --model_save_path "$MODEL" \
  --enable_ood \
  --motion_score_fn_path "$SCORE_FN" \
  --conformal_calibrator "$CALIB" \
  --conformal_calibrator_max "$CALIB_MAX" \
  --conformal_calibrator_uncalibrated "$CALIB_UNC" \
  --output_dir "$OUTPUT_DIR"

# 3) OOD detection performance of the two sketched-Lanczos detectors (AUROC, plus TPR/FPR/TNR/FNR
#    at the deployed thresholds: pose OOD_THRESHOLD from pose_estimation/h36m_settings.py, motion
#    OOD_THRESHOLD from motion_prediction/h36m_settings.py). These are separate ID-vs-OOD runs,
#    each against its own OOD set -- pose: H36M (ID) vs tiger-pose (OOD); motion: H36M validation
#    (ID) vs the same windows with the inputs permuted in time (OOD). Each run is skipped when its
#    metrics JSON already carries the rates; delete the JSON to force a re-run.
POSE_OOD_DIR="${POSE_OOD_DIR:-results/pose_prediction_ood}"
MOTION_OOD_DIR="${MOTION_OOD_DIR:-results/motion_prediction_ood_randproj}"
POSE_OOD_METRICS="$POSE_OOD_DIR/pose_ood_detection_metrics.json"
MOTION_OOD_METRICS="$MOTION_OOD_DIR/$(basename "$SCORE_FN" .cloudpickle)_results.json"

if ! grep -q '"tpr"' "$POSE_OOD_METRICS" 2>/dev/null; then
  echo "==================== pose OOD detection (H36M vs tiger-pose) ===================="
  python -m conformal_human_motion_prediction.examples.id_vs_ood_pose_prediction \
    --output_dir "$POSE_OOD_DIR" \
    --max_samples 500 \
    --h36m_max_files 4 \
    --h36m_frames_per_sequence 5
fi
if ! grep -q '"tpr"' "$MOTION_OOD_METRICS" 2>/dev/null; then
  echo "==================== motion OOD detection (H36M vs time-shuffled) ===================="
  # A finished run saves its raw scores, so the rates can be re-derived exactly without touching
  # the GPU again; only a missing scores file needs the full 2 x 10 000-sample inference pass.
  MOTION_OOD_SCORES="$MOTION_OOD_DIR/$(basename "$SCORE_FN" .cloudpickle)_ood_scores.cloudpickle"
  if [ -f "$MOTION_OOD_SCORES" ]; then
    python -m conformal_human_motion_prediction.examples.id_vs_ood_motion_prediction \
      --load_scores "$MOTION_OOD_SCORES" \
      --output_dir "$MOTION_OOD_DIR"
  else
    python -m conformal_human_motion_prediction.examples.id_vs_ood_motion_prediction \
      --score_fn "$SCORE_FN" \
      --max_samples 10000 \
      --seed 0 \
      --output_dir "$MOTION_OOD_DIR"
  fi
fi

# 4) Build the standalone conformal prediction-set LaTeX table (coverage/volume per method, plus
#    the OOD detection table from the two metrics JSONs above).
python -m conformal_human_motion_prediction.generate_plots.generate_conformal_prediction_set_results \
  --results_dir "$OUTPUT_DIR" \
  --pose_ood_metrics "$POSE_OOD_METRICS" \
  --motion_ood_metrics "$MOTION_OOD_METRICS"

#!/bin/bash
# Table 3/3: the combined final results table (tab:all_conformal_results) fusing both halves:
#   * conformal prediction-set coverage / volume  (from motion_prediction_conformal_prediction_set_results.sh)
#   * robot-shield certification columns           (from robot_shield_safety_results.sh)
#
# This only combines the CSVs the two upstream scripts already produced -- run them first:
#   ./final_results/motion_prediction_conformal_prediction_set_results.sh
#   ./final_results/robot_shield_safety_results.sh
set -e

COVERAGE_DIR="${COVERAGE_DIR:-results/final/conformal_prediction_sets}"
SHIELD_CSV="${SHIELD_CSV:-results/final/robot_shield/shield_results.csv}"
OUTPUT="${OUTPUT:-results/final/all_conformal_results.tex}"

python -m conformal_human_motion_prediction.generate_plots.generate_full_conformal_prediction_results_table \
  --coverage_dir "$COVERAGE_DIR" \
  --shield_csv "$SHIELD_CSV" \
  --output "$OUTPUT" \
  --confidence 0.9999

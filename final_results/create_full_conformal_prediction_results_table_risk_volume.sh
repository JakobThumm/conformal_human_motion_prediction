#!/bin/bash
# Table 3/3, volume-estimator variant: the combined final results table
# (tab:all_conformal_results_risk_volume) fusing both halves:
#   * conformal prediction-set coverage / volume  (from motion_prediction_conformal_prediction_set_results.sh)
#   * volume-weighted certification columns        (from robot_shield_safety_results_risk_volume.sh)
#
# Same left half as create_full_conformal_prediction_results_table.sh. The right half differs: that
# script reports the NAIVE shield simulation's pooled counts (c_safe, c_safe & contact, PFH_D over
# N dependent test cycles), whereas this one reports the factorisation the volume estimator
# measures,
#
#   PFH_D <= N_h * P_up(F) * [V(W)/V(F)] * P_up(D | F, W),
#
# i.e. P(F) (data-limited, deflated to n_eff = N_F / (2 L_corr) for window overlap), the
# closed-form witness-region volume ratio, and k_D dangerous trials out of N_D i.i.d. draws from W.
# That shows WHERE each method's risk comes from instead of one pooled count, and its binomial
# bound is exact rather than cluster-collapsed.
#
# This only combines the CSVs the two upstream scripts already produced -- run them first:
#   ./final_results/motion_prediction_conformal_prediction_set_results.sh
#   ./final_results/robot_shield_safety_results_risk_volume.sh
#
# Knobs:
#   COVERAGE_DIR  coverage CSV directory
#   RISK_CSV      CSV written by simulate_shield_failure_risk_volume --results_csv
#   OUTPUT        output .tex path
#   METHODS       row selection (see below)
set -e

COVERAGE_DIR="${COVERAGE_DIR:-results/final/conformal_prediction_sets}"
RISK_CSV="${RISK_CSV:-results/final/robot_shield_risk_volume/shield_risk_volume_results.csv}"
OUTPUT="${OUTPUT:-results/final/all_conformal_results_risk_volume.tex}"
# The five rows planned for the paper, in canonical row order. The two remaining ablation cells
# (alpha_max / no-calib. with OOD *inputs*) are excluded even if their coverage CSVs or risk rows
# exist; set METHODS=all to show everything.
METHODS="${METHODS:-iso_no_ood ours_no_ood ours_uncal_ood ours_max_ood ours_ood}"

for f in "$RISK_CSV"; do
  if [ ! -f "$f" ]; then
    echo "missing prerequisite: $f" >&2
    echo "  run ./final_results/robot_shield_safety_results_risk_volume.sh first" >&2
    exit 1
  fi
done

python -m conformal_human_motion_prediction.generate_plots.generate_full_risk_volume_results_table \
  --coverage_dir "$COVERAGE_DIR" \
  --risk_csv "$RISK_CSV" \
  --output "$OUTPUT" \
  --methods "$METHODS"

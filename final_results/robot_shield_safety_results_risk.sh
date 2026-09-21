#!/bin/bash
# Table 2/3 (factorised estimator): robot-shield dangerous-failure rate on H36M via
#   PFH_D = N_h * P(F) * P(D | F),      N_h = 3600 s / t_cycle
# i.e. examples.simulate_shield_failure_risk instead of examples.simulate_robot_shield.
#
# Why a second script. robot_shield_safety_results.sh counts the rare event directly over
# N ~= 2e13 (pose, trajectory, window) cycles and puts a Clopper-Pearson bound on it. Those cycles
# reuse the same ~6e4-window motion corpus ~1e8 times, so the binomial N is not an independent
# sample count. This script never counts the rare event over dependent cycles. It splits it:
#
#   * P(F)     -- prediction failure (the truth escapes the predicted set at some horizon step and
#                 joint). Measured on the recorded test windows: DATA-limited.
#   * P(D | F) -- that a failed prediction actually produces a verified-but-contact event. Measured
#                 by replaying ONLY the failing windows against random human placements and robot
#                 trajectory phases. Those placements are i.i.d. draws we generate, so this factor
#                 is COMPUTE-limited and its binomial bound is legitimate.
#
# The factorisation is exact because a dangerous failure implies a prediction failure (if the truth
# stayed inside the predicted occupancy, an empty intersection in the verification condition
# excludes contact). --verify_lemma tests that empirically on non-failure windows.
#
# RUN ORDER IS A PRIORITY ORDER, matching robot_shield_safety_results.sh, so an interrupted sweep
# still leaves the rows the paper needs most. The CSV is appended to by default.
#   1) Ours OOD filtered            4) Ours (no calib.) OOD filtered
#   2) ISO 13855                    5) Ours with OOD inputs
#   3) Ours (alpha_max) OOD filtered
#
# Knobs:
#   FRESH=1            wipe $CSV first (default 0: append)
#   RUN_OPTIONAL=1     also run the two OOD-inputs ablation cells (default 0)
#   SMOKE=1            tiny run (1e5 placements, 32 failures) to check the plumbing, ~1 min
#   NUM_ROBOT_POSES    placements per method (see SIZING below)
#   SHARDS             worker processes over the placement integral (measured 1.22x at 4; the GPU
#                      saturates, 8/16 were worse)
#   GPU_A_CHUNK        active motions per GPU kernel launch. MUST be retuned per method: every
#                      launch is padded to this width, and the optimum is ~1.5x the mean active-
#                      motion count at the nearest power of two. 128 was measured best at
#                      M = 271 failing windows; the ISO and no-calibration rows have a far higher
#                      miss-rate and therefore many more failing windows, so raise it there. Each
#                      run prints the observed counts and a suggestion.
#
# SIZING -- how NUM_ROBOT_POSES follows from the exposure target.
# The deliverable is inverted into an exposure budget: the share of an operating hour of
# human-in-workspace exposure that keeps PFH_D under the PL d line of 1e-6/h. A budget of at least
# T seconds per hour needs
#
#   lambda_d <= 1e-6 * 3600 / T,   lambda_d = P(F) * N_h * P(D|F)_upper
#
# and with zero dangerous placements observed the one-sided Clopper-Pearson limit at confidence
# 1-eps is P(D|F)_upper ~= ln(1/eps) / N_D, so
#
#   N_D >= P(F) * N_h * ln(1/eps) * T / (1e-6 * 3600).
#
# Measured on the H36M VALIDATION split (P_up(F) = 1.36e-2 at eps_F = 5e-6, N_h = 9e5), so
# PFH_D <= 1.23e4 * P_up(D|F) and the placement requirement is
#
#     T = 1 s/h  ->  N_D >= 4.2e7   (~1.7 h/method at 7e3 placements/s)
#     T = 2 s/h  ->  N_D >= 8.3e7   (~3.3 h/method)
#     T = 5 s/h  ->  N_D >= 2.1e8   (~8.3 h/method)
#     T = 10 s/h ->  N_D >= 4.2e8   (~17 h/method)
#
# The default below is sized for T = 1 s/h with ~10% headroom. P(F) on the TEST split is only
# known after the run, which prints it along with the achieved budget: if the budget lands under
# the target, scale NUM_ROBOT_POSES by (target / achieved) and re-run. Throughput was ~5.7e3
# placements/s at GPU_A_CHUNK=128 with 271 failing windows and ~7e3/s with SHARDS=4; the ISO 13855
# row has a far higher miss-rate, hence more failing windows, hence fewer placements/s.
#
# Prerequisites (NOT run here): the calibrator .npz files and the test-set predictions, both
# produced by robot_shield_safety_results.sh. Run that first, or with SKIP_CALIB=1 if the
# calibrators already exist.
set -e

# export XLA_PYTHON_CLIENT_PREALLOCATE=false   # share the GPU politely (jax pre-alloc off)

CALIB="models/motion_prediction/conformal_calibration/conformal_calibrator.npz"
CALIB_MAX="models/motion_prediction/conformal_calibration/conformal_calibrator_max.npz"
CALIB_UNC="models/motion_prediction/conformal_calibration/conformal_calibrator_uncalibrated.npz"
RESULTS_TEST="results/motion_prediction/motion_prediction_results_test.cloudpickle"
OUT_DIR="results/final/robot_shield_risk"
CSV="$OUT_DIR/shield_risk_results.csv"

TARGET_S_PER_H="${TARGET_S_PER_H:-1}"          # exposure-budget target, documented above
NUM_ROBOT_POSES="${NUM_ROBOT_POSES:-46000000}" # placements per method (see SIZING)
SHARDS="${SHARDS:-4}"
GPU_A_CHUNK="${GPU_A_CHUNK:-128}"
POSE_RADIUS="${POSE_RADIUS:-10.0}"
# P(F) is always the all-window rate k_F / |Z_test| (the paper's estimator). Window dependence is
# handled on the INTERVAL instead, by deflating the window count to an effective sample size
#   n_eff = N_F / (2 * L_corr)
# where L_corr is the integrated autocorrelation time of the failure indicator (measured by Sokal
# windowing; override with L_CORR for a sensitivity check) and the 2 accounts for the loader
# emitting both 50->25 fps phase offsets. FAILURE_STRIDE is therefore only a diagnostic: the run
# prints the strided rate beside the all-window one, and a large gap warns that the stride landed
# on an unlucky phase. Leave it at 1 unless you are running the striding experiment.
FAILURE_STRIDE="${FAILURE_STRIDE:-1}"
L_CORR="${L_CORR:-}"                           # empty = measure it
FRESH="${FRESH:-0}"
RUN_OPTIONAL="${RUN_OPTIONAL:-0}"
SMOKE="${SMOKE:-0}"

if [ "$SMOKE" != "0" ]; then
  NUM_ROBOT_POSES=100000
  SHARDS=1
  EXTRA_SMOKE="--max_failures_eval 32"
  echo "### SMOKE run: $NUM_ROBOT_POSES placements, 32 failures -- plumbing check only,"
  echo "### the bounds it prints are NOT reportable."
else
  EXTRA_SMOKE=""
fi

mkdir -p "$OUT_DIR"
if [ "$FRESH" != "0" ]; then
  rm -f "$CSV"
fi

for f in "$RESULTS_TEST" "$CALIB" "$CALIB_MAX" "$CALIB_UNC"; do
  if [ ! -f "$f" ]; then
    echo "missing prerequisite: $f (run final_results/robot_shield_safety_results.sh first)" >&2
    exit 1
  fi
done

echo "Exposure-budget target: $TARGET_S_PER_H s/h   placements per method: $NUM_ROBOT_POSES"

COMMON="--results_file $RESULTS_TEST \
  --backend gpu --gpu_dtype float32 --gpu_a_chunk $GPU_A_CHUNK \
  --num_robot_poses $NUM_ROBOT_POSES --shards $SHARDS \
  --pose_radius $POSE_RADIUS --pose_z_offset 0.2 \
  --robot_stride 25 --failure_stride $FAILURE_STRIDE --seed 0 \
  --results_csv $CSV $EXTRA_SMOKE ${L_CORR:+--l_corr $L_CORR}"

run_risk () {                         # $1 = banner, $2 = tag for the per-failure dump, rest = flags
  local banner="$1"; local tag="$2"; shift 2
  echo "==================== $banner ===================="
  python -m conformal_human_motion_prediction.examples.simulate_shield_failure_risk \
    $COMMON --save_per_failure "$OUT_DIR/per_failure_${tag}.npz" "$@"
}

# --verify_lemma only on the first run: it is a property of the geometry, not of the method, and
# it costs a second replay pass over non-failing windows.
run_risk "1/5  Ours OOD filtered" ours_ood \
  --conformal_calibrator "$CALIB" --human_set conformal --mask_ood --verify_lemma 20000

run_risk "2/5  ISO 13855 (SARA, v=2 m/s)" iso_no_ood \
  --human_set sara --no-mask_ood

run_risk "3/5  Ours (alpha_max ablation) OOD filtered" ours_max_ood \
  --conformal_calibrator "$CALIB_MAX" --human_set conformal --mask_ood

run_risk "4/5  Ours (no calibration ablation) OOD filtered" ours_uncal_ood \
  --conformal_calibrator "$CALIB_UNC" --human_set conformal --mask_ood

run_risk "5/5  Ours with OOD inputs" ours_no_ood \
  --conformal_calibrator "$CALIB" --human_set conformal --no-mask_ood

if [ "$RUN_OPTIONAL" != "0" ]; then   # not planned for the paper
  run_risk "opt  Ours (alpha_max ablation) with OOD inputs" ours_max_no_ood \
    --conformal_calibrator "$CALIB_MAX" --human_set conformal --no-mask_ood

  run_risk "opt  Ours (no calib. ablation) with OOD inputs" ours_uncal_no_ood \
    --conformal_calibrator "$CALIB_UNC" --human_set conformal --no-mask_ood
fi

echo
echo "Per-method rows appended to $CSV"
echo "Per-failure escape sizes / P(d|f_i) in $OUT_DIR/per_failure_*.npz"

#!/bin/bash
# Table 2/3 (volume-weighted estimator): robot-shield dangerous-failure rate on H36M via
#   PFH_D = N_h * P(F) * P(W|F) * P(D | F, W),      N_h = 3600 s / t_cycle
# i.e. examples.simulate_shield_failure_risk_volume instead of ..._risk.
#
# Why a third script. robot_shield_safety_results_risk.sh already factorises the rare event into
# the DATA-limited P(F) and the COMPUTE-limited P(D|F). Its placement integral is still a rejection
# sampler, though: placements are drawn uniformly over the whole cell and ~88 % are thrown away by
# the level-1 gate before the kernel runs, and because one placement is applied to ALL failing
# windows x trajectories, its trials are a CLUSTER -- so the bound has to collapse a placement to
# "was ANY window dangerous here?", which costs orders of magnitude of resolution.
#
# This script removes both losses at once. A trial is one verification cycle: a triple of a failing
# window m, a phase j of the long-horizon robot trajectory, and a base placement (yaw, t). Trials
# are drawn DIRECTLY from the witness region W -- the shield's own level-4 bounding-sphere test,
# hence a necessary condition for contact, hence D => W -- and the price is the volume ratio
#
#   P(W|F) = V(W)/V(F) = mean over (m, j) of vol(level-4 ball n cylinder) / vol(cylinder),
#
# which is CLOSED FORM (the ball's z-offset is yaw-invariant, so it is a spherical-band integral),
# not estimated: no Monte-Carlo error enters that factor. The trials are then genuinely i.i.d., so
# their Clopper-Pearson bound is exact rather than a cluster-collapsed one, and the level-5 test
# resolves ~74 % of them as provable zeros for free (they still count in N_W).
#
# Measured on an RTX 5090: ~4.2e6 trials/s = ~2.5e8 equivalent uniform placements/s, versus ~7e3
# placements/s for the cross-product script. The P(F) side -- point estimate k_F/|Z_test|, interval
# deflated to n_eff = N_F/(2 L_corr) by the integrated autocorrelation time, one-sided
# Clopper-Pearson, symmetric eps split -- is the SAME code as ..._risk.sh (imported, not copied),
# so the two estimators' data factors are directly comparable and only the placement factor moves.
#
# RUN ORDER IS A PRIORITY ORDER, matching the other two scripts, so an interrupted sweep still
# leaves the rows the paper needs most. The CSV is appended to by default.
#   1) Ours OOD filtered            4) Ours (no calib.) OOD filtered
#   2) ISO 13855                    5) Ours with OOD inputs
#   3) Ours (alpha_max) OOD filtered
#
# Knobs:
#   FRESH=1            wipe $CSV first (default 0: append)
#   RUN_OPTIONAL=1     also run the two OOD-inputs ablation cells (default 0)
#   SMOKE=1            tiny run (2e7 trials, 32 failures) to check the plumbing, ~2 min
#   NUM_TRIALS         N_W per method: trials drawn from W (see SIZING below)
#   EPSILON_D          joint error budget of the reported bound (default 1e-5 = 99.999 %)
#   L_CORR             override the measured integrated autocorrelation time (sensitivity check)
#   SELF_TESTS=0       skip the float64 self-test pass that runs before the five methods
#
# SIZING -- how NUM_TRIALS follows from the PFH_D target.
# NUM_TRIALS is N_W, the number of trials drawn FROM the witness region W -- not the uniform
# placement count N_D of eq. (5)'s reference measure, which is the larger N_W / P(W|F) (the run
# prints it). The Clopper-Pearson interval bounds the conditional P(D | F, W) and the closed-form
# volume ratio supplies the rest, so with ZERO dangerous trials observed
# P_up(D|F,W) ~= ln(1/eps) / N_W and the bound clears a target PFH_D when
#
#   N_W >= N_h * P_up(F) * P(W|F) * ln(1/eps) / PFH_D_target.
#
# With N_h = 9e5 (t_cycle = 4 ms), P_up(F) ~= 1.3e-2 (test split, OOD filtered, after the n_eff
# deflation), P(W|F) = 1.5e-2, eps = eps_(D|F) = 5e-6 (ln = 12.2) and a target of 1e-6 this is
# N_W >= 2.2e9 -- hence the 4e9 default, ~1.8x that, running ~15 min per method.
#
# CAVEAT: that formula holds only at k_D = 0. The 4e9 run observed k_D = 4, and the CP limit at
# k = 4 is 1.76x the zero-event one, which consumed the whole 1.8x margin: the realised bound was
# 9.50e-7, only 5 % under 1e-6. Once events appear, k_D grows proportionally to N_W and the bound
# converges to the point estimate instead of falling as 1/N_W (4e9 -> 9.5e-7, 4e10 -> 3.3e-7,
# asymptote ~1.8e-7), so budget 4e10 if the headline needs real margin. P_up(F) is only known
# after the run, which prints every factor; the ISO and no-calibration rows have a far higher
# miss-rate, hence a larger P_up(F), and need proportionally more trials.
#
# Prerequisites (NOT run here): the calibrator .npz files and the test-set predictions, both
# produced by robot_shield_safety_results.sh. Run that first, or with SKIP_CALIB=1 if the
# calibrators already exist.
set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false   # share the GPU politely (jax pre-alloc off)

CALIB="models/motion_prediction/conformal_calibration/conformal_calibrator.npz"
CALIB_MAX="models/motion_prediction/conformal_calibration/conformal_calibrator_max.npz"
CALIB_UNC="models/motion_prediction/conformal_calibration/conformal_calibrator_uncalibrated.npz"
RESULTS_TEST="results/motion_prediction/motion_prediction_results_test.cloudpickle"
OUT_DIR="results/final/robot_shield_risk_volume"
CSV="$OUT_DIR/shield_risk_volume_results.csv"

TARGET_S_PER_H="${TARGET_S_PER_H:-3600}"       # exposure-budget target (3600 = full exposure)
NUM_TRIALS="${NUM_TRIALS:-4000000000}"         # N_W per method (see SIZING)
EPSILON_D="${EPSILON_D:-1e-5}"
POSE_RADIUS="${POSE_RADIUS:-10.0}"
# A trial uses ONE trajectory phase, so there is no reason to stride the robot grid: stride 1 puts
# the full 4 ms planning grid (2358 phases) in the phase distribution at zero extra cost.
ROBOT_STRIDE="${ROBOT_STRIDE:-1}"
# Host-side sampler batch / GPU kernel batch. Measured optimum on a 5090; the run is sampler-bound,
# so the kernel batch is flat from 4k to 64k.
TRIAL_BLOCK="${TRIAL_BLOCK:-4000000}"
KERNEL_BATCH="${KERNEL_BATCH:-16384}"
L_CORR="${L_CORR:-}"                           # empty = measure it
SELF_TESTS="${SELF_TESTS:-1}"
FRESH="${FRESH:-0}"
RUN_OPTIONAL="${RUN_OPTIONAL:-0}"
SMOKE="${SMOKE:-0}"

if [ "$SMOKE" != "0" ]; then
  NUM_TRIALS=20000000
  EXTRA_SMOKE="--max_failures_eval 32"
  echo "### SMOKE run: $NUM_TRIALS trials, 32 failures -- plumbing check only,"
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

echo "Exposure-budget target: $TARGET_S_PER_H s/h   trials per method: $NUM_TRIALS"

COMMON="--results_file $RESULTS_TEST \
  --backend gpu --gpu_dtype float32 \
  --num_trials $NUM_TRIALS --trial_block $TRIAL_BLOCK --kernel_batch $KERNEL_BATCH \
  --pose_radius $POSE_RADIUS --pose_z_offset 0.2 \
  --robot_stride $ROBOT_STRIDE --epsilon_d $EPSILON_D --seed 0 \
  --results_csv $CSV $EXTRA_SMOKE ${L_CORR:+--l_corr $L_CORR}"

run_risk () {                         # $1 = banner, $2 = tag for the per-trial dump, rest = flags
  local banner="$1"; local tag="$2"; shift 2
  echo "==================== $banner ===================="
  python -m conformal_human_motion_prediction.examples.simulate_shield_failure_risk_volume \
    $COMMON --save_trials "$OUT_DIR/trials_${tag}.npz" "$@"
}

# The three self-tests of the chain, run ONCE before the methods and in float64:
#   --parity        the per-trial kernel vs run_pose_pure over every (window, trajectory) pair
#   --verify_gate   D => W: no uniformly drawn CONTACT may fall outside the witness region
#   --verify_lemma  D => F: zero verified-AND-contact on non-failure windows
# They are properties of the geometry and the kernel, not of a method, so they do not need the
# production trial count -- and --parity is only zero-tolerance in float64: in float32 a pair
# within ~1e-6 m of exact tangency can land on either side of the `<=`, which the check reports as
# a ~1e-4 relative deviation rather than a failure. The production runs below stay in float32,
# where that rounding cannot move k_D (it is two orders of magnitude below any contact count and
# never touched the verified-AND-contact set in testing).
if [ "$SELF_TESTS" != "0" ]; then
  echo "==================== 0/5  Self-tests (float64) ===================="
  python -m conformal_human_motion_prediction.examples.simulate_shield_failure_risk_volume \
    --results_file "$RESULTS_TEST" --conformal_calibrator "$CALIB" \
    --human_set conformal --mask_ood \
    --backend gpu --gpu_dtype float64 --robot_stride "$ROBOT_STRIDE" \
    --pose_radius "$POSE_RADIUS" --pose_z_offset 0.2 --epsilon_d "$EPSILON_D" --seed 0 \
    --num_trials 20000000 --trial_block "$TRIAL_BLOCK" --kernel_batch "$KERNEL_BATCH" \
    --parity 2 --verify_gate 5000000 --verify_lemma 20000000 --lemma_windows 4000 \
    ${L_CORR:+--l_corr $L_CORR} 2>&1 | tee "$OUT_DIR/self_tests.log" | \
    grep -E "^  pose |^PARITY|^  D => W|^  contacts OUTSIDE|^LEMMA|^!!"
  echo "    full self-test output in $OUT_DIR/self_tests.log"
fi

run_risk "1/5  Ours OOD filtered" ours_ood \
  --conformal_calibrator "$CALIB" --human_set conformal --mask_ood

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
echo "Per-(window, trajectory) volume weights / dangerous trials in $OUT_DIR/trials_*.npz"
if [ "$SELF_TESTS" != "0" ]; then
  echo "Self-test log (parity / D => W / D => F) in $OUT_DIR/self_tests.log"
fi

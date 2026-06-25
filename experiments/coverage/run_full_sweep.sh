#!/bin/bash
# Full coverage-improvement sweep: retrain Stage 4 (40 epochs, batch 256, from the canonical Stage-3
# checkpoint) for the control and each P1/P2/P4 variant, then regenerate the validation cloudpickle
# and run the scoreboards. Each variant differs from the control ONLY by its trailing flags.
#
# Robust for overnight/unattended use: a single variant failing does NOT abort the rest. Results land
# in results/coverage_experiments/<RUN_ID>/. We only READ models/motion_prediction/final_training_run
# (Stage-3 init); nothing here writes to any canonical artifact dir (run_variant.sh refuses those IDs).
#
# Ordered most-informative-first in case the shared GPU gets reclaimed mid-sweep.
cd "$(dirname "$0")/../.."
RUN=experiments/coverage/run_variant.sh

STAGE3="models/motion_prediction/final_training_run/checkpoints/stage_3/dct_pose_transformer.pickle"
if [ ! -f "$STAGE3" ]; then echo "Missing Stage-3 init pickle: $STAGE3"; exit 1; fi

run() {  # run() <RUN_ID> [flags...] ; never aborts the sweep
  local id="$1"; shift
  echo "######## $(date) starting $id ########"
  if bash "$RUN" "$id" "$@"; then
    echo "######## $(date) OK $id ########"
  else
    echo "######## $(date) FAILED $id (rc=$?), continuing ########"
  fi
}

# Control: identical pipeline, no coverage flags -> reproduces deployed final_model, isolates retrain noise.
run cov_control
# P2: pinball/quantile loss on the deployed set radius (targets M1 + M2).
run cov_p2_pinball    --lambda_pinball 1.0
# P2 + P4: add tail reweighting by input uncertainty (the recommended set).
run cov_p2p4          --lambda_pinball 1.0 --tail_reweight_gamma 1.0 --tail_reweight_max 5.0
# P2(strong) + P4: stronger pinball so the model self-calibrates (less reliance on post-hoc affine).
run cov_p2hi_p4       --lambda_pinball 3.0 --tail_reweight_gamma 1.0 --tail_reweight_max 5.0

echo "SWEEP COMPLETE $(date). Summaries: results/coverage_experiments/*/coverage_failures/per_joint_failures.csv"

#!/bin/bash
# Train one Stage-4 motion-prediction variant from the canonical Stage-3 checkpoint, regenerate the
# VALIDATION results cloudpickle, and run the coverage scoreboard. The only thing that should differ
# between variants is the trailing P1/P2/P4 flags -- every other hyperparameter matches the deployed
# `final_model` run (see models/motion_prediction/final_training_run/dct_pose_transformer_args.json).
#
# Usage:
#   experiments/coverage/run_variant.sh <RUN_ID> [extra train flags...]
# e.g.
#   experiments/coverage/run_variant.sh p2_pinball  --lambda_pinball 1.0
#   experiments/coverage/run_variant.sh p2p4        --lambda_pinball 3.0 --tail_reweight_gamma 1.0
#
# Env overrides (defaults reproduce the full final_model setup):
#   EPOCHS (40)  BATCH (256)  NSAMPLES (unset=full; set for a quick subsampled proof)  SKIP_TRAIN (0)
set -e
cd "$(dirname "$0")/../.."

RUN_ID="$1"; shift || true
if [ -z "$RUN_ID" ]; then echo "usage: run_variant.sh <RUN_ID> [train flags...]"; exit 1; fi
# Safety: never write into the canonical artifacts. We only READ Stage-3 from final_training_run.
case "$RUN_ID" in
  final_training_run|final_model|final_model_for_ood|old_models)
    echo "REFUSING to use RUN_ID='$RUN_ID' -- that is a canonical artifact dir."; exit 1;;
esac
EXTRA_FLAGS="$@"

EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-256}"
STAGE3_PICKLE="models/motion_prediction/final_training_run/checkpoints/stage_3/dct_pose_transformer.pickle"
MODEL_PICKLE="models/motion_prediction/${RUN_ID}/checkpoints/stage_4/dct_pose_transformer.pickle"
RESULT_DIR="results/coverage_experiments/${RUN_ID}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
NS_FLAG=""
if [ -n "$NSAMPLES" ]; then NS_FLAG="--n_samples ${NSAMPLES}"; fi

echo "============================================================"
echo "VARIANT ${RUN_ID}  |  epochs=${EPOCHS} batch=${BATCH} nsamples=${NSAMPLES:-full}"
echo "extra flags: ${EXTRA_FLAGS:-<none (control)>}"
echo "============================================================"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  .venv/bin/python -m conformal_human_motion_prediction.motion_prediction.train_motion_prediction_model \
    --run_id "${RUN_ID}" --stage 4 --new_config --no_wandb \
    --init_weights_path "${STAGE3_PICKLE}" \
    --data_path datasets/ \
    --d_model 128 --nhead 4 --num_layers 2 --seq_len 50 --seq_len_output 10 \
    --batch_size "${BATCH}" --learning_rate 1e-4 --weight_decay 1e-6 --max_grad_norm 0.68 \
    --use_lr_schedule --lr_schedule_type cosine --lr_warmup_epochs 3 --lr_min_factor 0.1 \
    --augment --stage4_epochs "${EPOCHS}" --seed 420 \
    ${NS_FLAG} ${EXTRA_FLAGS}
fi

echo "--- Regenerating VALIDATION results cloudpickle from ${MODEL_PICKLE} ---"
.venv/bin/python -m conformal_human_motion_prediction.examples.motion_prediction \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3DWithInputUncertainty \
  --split validation \
  --model_save_path "${MODEL_PICKLE}" \
  --output_dir "${RESULT_DIR}"

echo "--- Coverage scoreboard ---"
.venv/bin/python -m conformal_human_motion_prediction.motion_prediction.evaluate_covariance_failures \
  --results_file "${RESULT_DIR}/motion_prediction_results_validation.cloudpickle" \
  --output_dir "${RESULT_DIR}/coverage_failures"

echo "--- Overall coverage / volume / MPJPE ---"
.venv/bin/python -m conformal_human_motion_prediction.motion_prediction.evaluate_covariance \
  --results_file "${RESULT_DIR}/motion_prediction_results_validation.cloudpickle" \
  --output_dir "${RESULT_DIR}/coverage_overall"

echo "DONE variant ${RUN_ID} -> ${RESULT_DIR}"

#!/usr/bin/env python3
"""Build the deployable motion-prediction models from a training run.

Given a `final_training_run/` produced by
`motion_prediction/train_motion_prediction_model.py` (which writes, per stage, an Orbax
checkpoint plus an exported `dct_pose_transformer.pickle` + `dct_pose_transformer_args.json`),
this derives the artifacts the eval/OOD pipeline expects:

  * `final_model/`         -- the deployable full model (a copy of the final stage export).
  * `final_model_for_ood/` -- the DEFAULT OOD model: the same weights plus a frozen random
                              orthonormal projection of the 390-dim future motion onto
                              OOD_PROJECTION_DIM directions, and an args.json selecting
                              `DCTPoseTransformerRandomProjection`.
  * `final_model_for_ood_fixed_joints/` -- the legacy alternative: the same weights with an
                              args.json selecting `DCTPoseTransformerReducedOutput` (output_dim
                              = 9: timestep REDUCED_TIMESTEP, joints REDUCED_JOINT_INDICES).
                              That reduction is pure model-class behaviour, so this needs no
                              extra parameters at all.

Both heads score ~0.982 AUROC on the time-shuffled H36M OOD benchmark; the projection is the
default because it removes an arbitrary design choice (9 hand-picked coordinates out of 390) at
no cost in detection performance. See docs/RESULTS.md, section "OOD Detection".

The OOD *score function* is built separately by running `ood_scoring.score_model` against one of
these directories (see README section 2.3).

Usage:
    python scripts/build_motion_models.py \
        --run_dir models/motion_prediction/final_training_run \
        --output_root models/motion_prediction
"""
import argparse
import json
import os
import pickle
import shutil

import jax
import jax.numpy as jnp
import numpy as np

from conformal_human_motion_prediction.models.wrapper import model_from_string
from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
    INPUT_HORIZON_LENGTH,
    N_JOINTS,
    OOD_PROJECTION_DIM,
    OOD_PROJECTION_SEED,
    REDUCED_JOINT_INDICES,
)

PICKLE_NAME = "dct_pose_transformer.pickle"
ARGS_NAME = "dct_pose_transformer_args.json"

# Default OOD model: a frozen random orthonormal projection over the whole future motion.
PROJECTION_MODEL_NAME = "DCTPoseTransformerRandomProjection"
PROJECTION_PARAM_KEY = "random_projection"

# Legacy OOD model: one timestep, REDUCED_JOINT_INDICES joints, 3 coords each.
REDUCED_OUTPUT_DIM = len(REDUCED_JOINT_INDICES) * 3
REDUCED_MODEL_NAME = "DCTPoseTransformerReducedOutput"

OOD_DATASET = "Human36mMotionReducedOutputDataset3DAugmented"


def _latest_stage(checkpoints_dir):
    """Return the highest-numbered stage_N dir that has an exported pickle."""
    stages = []
    for name in os.listdir(checkpoints_dir):
        if name.startswith("stage_") and name[len("stage_"):].isdigit():
            if os.path.isfile(os.path.join(checkpoints_dir, name, PICKLE_NAME)):
                stages.append(int(name[len("stage_"):]))
    if not stages:
        raise FileNotFoundError(
            f"No stage_*/{PICKLE_NAME} found under {checkpoints_dir}. "
            f"Run motion training (it writes per-stage pickles) before building."
        )
    return max(stages)


def _to_plain(tree):
    """flax FrozenDict / dict -> nested plain dicts of numpy arrays."""
    if hasattr(tree, "items"):
        return {k: _to_plain(v) for k, v in tree.items()}
    return np.asarray(tree)


def make_projection_params(projection_dim, seed):
    """Draw the frozen orthonormal projection by initialising the OOD model.

    The matrix is stored in the built pickle, so scoring never depends on re-drawing it; the
    seed only matters at build time. Changing it invalidates any existing score function.
    """
    model = model_from_string(PROJECTION_MODEL_NAME, projection_dim)
    dummy = jnp.zeros((1, INPUT_HORIZON_LENGTH, N_JOINTS * 3))
    params = model.init(jax.random.key(seed), dummy)["params"]
    projection = _to_plain(params[PROJECTION_PARAM_KEY])
    kernel = projection["kernel"]
    deviation = float(np.abs(kernel.T @ kernel - np.eye(projection_dim)).max())
    print(f"  projection kernel {kernel.shape}, |Q^T Q - I| = {deviation:.2e} (orthonormal)")
    return projection


def _write_ood_dir(ood_dir, src_pickle, src_args, model_name, output_dim,
                   extra_params=None, readme=""):
    """Write an OOD model dir: the stage weights (+ optional head params) and a rewritten args.json."""
    os.makedirs(ood_dir, exist_ok=True)
    with open(src_pickle, "rb") as f:
        model_data = pickle.load(f)

    if extra_params:
        params = _to_plain(model_data["params"])
        for key, value in extra_params.items():
            if key in params:
                raise AssertionError(f"stage export already contains a '{key}' subtree")
            params[key] = value
        model_data = {**model_data, "params": params}
        with open(os.path.join(ood_dir, PICKLE_NAME), "wb") as f:
            pickle.dump(model_data, f)
    else:
        # No head parameters needed -- the reduction is pure model-class behaviour.
        shutil.copy2(src_pickle, os.path.join(ood_dir, PICKLE_NAME))

    with open(src_args) as f:
        args = json.load(f)
    args["model"] = model_name
    args["output_dim"] = output_dim
    args["dataset"] = OOD_DATASET
    with open(os.path.join(ood_dir, ARGS_NAME), "w") as f:
        json.dump(args, f, indent=2)
    with open(os.path.join(ood_dir, "README.md"), "w") as f:
        f.write(readme)
    print(f"Wrote {ood_dir}/  (model={model_name}, output_dim={output_dim})")


def build(run_dir, output_root, stage=None, heads=("random_projection", "fixed_joints"),
          projection_dim=OOD_PROJECTION_DIM, projection_seed=OOD_PROJECTION_SEED):
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"{checkpoints_dir} does not exist")

    stage = stage if stage is not None else _latest_stage(checkpoints_dir)
    stage_dir = os.path.join(checkpoints_dir, f"stage_{stage}")
    src_pickle = os.path.join(stage_dir, PICKLE_NAME)
    src_args = os.path.join(stage_dir, ARGS_NAME)
    if not os.path.isfile(src_pickle):
        raise FileNotFoundError(f"Missing {src_pickle}")
    if not os.path.isfile(src_args):
        raise FileNotFoundError(f"Missing {src_args}")
    print(f"Using stage_{stage} export from {stage_dir}")

    # 1) final_model/ -- verbatim copy of the final stage export.
    final_dir = os.path.join(output_root, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    shutil.copy2(src_pickle, os.path.join(final_dir, PICKLE_NAME))
    shutil.copy2(src_args, os.path.join(final_dir, ARGS_NAME))
    print(f"Wrote {final_dir}/  (full DCTPoseTransformer)")

    # 2) final_model_for_ood/ -- DEFAULT: frozen random orthonormal projection.
    if "random_projection" in heads:
        ood_dir = os.path.join(output_root, "final_model_for_ood")
        projection = make_projection_params(projection_dim, projection_seed)
        _write_ood_dir(
            ood_dir, src_pickle, src_args, PROJECTION_MODEL_NAME, projection_dim,
            extra_params={PROJECTION_PARAM_KEY: projection},
            readme=(
                "DEFAULT OOD model. The motion weights are identical to ../final_model; this dir\n"
                f"adds a frozen `{PROJECTION_PARAM_KEY}/kernel` of shape (390, {projection_dim}) with\n"
                f"orthonormal columns (seed {projection_seed}), and an args.json with\n"
                f"`model` -> {PROJECTION_MODEL_NAME}, `output_dim` -> {projection_dim}.\n\n"
                "The projection spans all 390 future-motion coordinates, so no output dimension is\n"
                "excluded by construction. Its parameters are stop-gradient-frozen and therefore do\n"
                "not enter the GGN.\n\n"
                "Generated by scripts/build_motion_models.py. Build the OOD score function by\n"
                "running ood_scoring.score_model against this directory.\n"
            ),
        )

    # 3) final_model_for_ood_fixed_joints/ -- legacy hand-picked coordinate slice.
    if "fixed_joints" in heads:
        fixed_dir = os.path.join(output_root, "final_model_for_ood_fixed_joints")
        _write_ood_dir(
            fixed_dir, src_pickle, src_args, REDUCED_MODEL_NAME, REDUCED_OUTPUT_DIM,
            readme=(
                "Legacy OOD model (alternative to the default ../final_model_for_ood). The weights\n"
                "are identical to ../final_model and no head parameters are added -- only the\n"
                f"args.json differs: `model` -> {REDUCED_MODEL_NAME}, `output_dim` ->\n"
                f"{REDUCED_OUTPUT_DIM} (timestep REDUCED_TIMESTEP, joints {REDUCED_JOINT_INDICES}).\n\n"
                "Generated by scripts/build_motion_models.py.\n"
            ),
        )

    print("\nNext: build the OOD score function with ood_scoring.score_model against the chosen\n"
          "OOD dir (see README section 2.3). Remember that OOD scores are NOT comparable across\n"
          "heads -- OOD_THRESHOLD in motion_prediction/h36m_settings.py is head-specific.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default="models/motion_prediction/final_training_run",
                   help="Training run dir containing checkpoints/stage_*/")
    p.add_argument("--output_root", default="models/motion_prediction",
                   help="Where to write final_model/ and the OOD model dirs")
    p.add_argument("--stage", type=int, default=None,
                   help="Stage to deploy (default: highest stage with an export)")
    p.add_argument("--ood_head", choices=("random_projection", "fixed_joints", "both"),
                   default="both",
                   help="Which OOD readout head(s) to build (default: both; the random "
                        "projection is the deployed default)")
    p.add_argument("--projection_dim", type=int, default=OOD_PROJECTION_DIM,
                   help=f"Output dim of the random projection (default: {OOD_PROJECTION_DIM})")
    p.add_argument("--projection_seed", type=int, default=OOD_PROJECTION_SEED,
                   help=f"Seed for the projection (default: {OOD_PROJECTION_SEED}). Changing it "
                        "invalidates existing score functions.")
    args = p.parse_args()
    heads = ("random_projection", "fixed_joints") if args.ood_head == "both" else (args.ood_head,)
    build(args.run_dir, args.output_root, args.stage, heads,
          args.projection_dim, args.projection_seed)


if __name__ == "__main__":
    main()

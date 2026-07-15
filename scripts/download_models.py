#!/usr/bin/env python3
"""Download the model checkpoints from the Hugging Face Hub into ``models/``.

Fetches the artifacts uploaded by ``scripts/upload_models.py`` and restores them to the
on-disk layout the pipeline expects, then derives the deployable motion models:

    models/
      pose_estimation/                              # JAX RegressFlow nets + camera params
        jax_resnet*_regressflow*_{args.json,params.pickle}
        camera-parameters.json
      motion_prediction/
        final_training_run/                         # per-stage Orbax checkpoints + exports
        final_model/                                # built from final stage (full model)
        final_model_for_ood/                        # built from final stage (reduced output)
        conformal_calibration/                      # fitted conformal_calibrator.npz
      ood_functions/                                # cached sketched-Lanczos OOD score fns
        {jax_resnet18_regressflow_3joints,dct_pose_transformer}_score_fn.cloudpickle

The two ``final_model*`` folders are NOT downloaded -- they are produced locally by
``scripts/build_motion_models.py`` from ``final_training_run`` so the reduced-output OOD
args stay in sync with the code (REDUCED_JOINT_INDICES).

Prerequisites:
    pip install -e .            # installs huggingface_hub
    # Public repo: no auth needed. Private repo: `hf auth login` or export HF_TOKEN.

Usage:
    python scripts/download_models.py                       # everything + build motion models
    python scripts/download_models.py --only pose_estimation
    python scripts/download_models.py --no-build            # skip building final_model*
    python scripts/download_models.py --repo_id you/your-repo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

DEFAULT_REPO_ID = "JakobThumm/conformal-human-motion-prediction-models"
REPO_TYPE = "model"

# Logical group -> glob patterns (relative to the repo root) to fetch for that group.
DOWNLOAD_GROUPS: dict[str, list[str]] = {
    "pose_estimation": ["pose_estimation/**"],
    "motion_prediction": ["motion_prediction/final_training_run/**"],
    "conformal_calibration": ["motion_prediction/conformal_calibration/**"],
    "ood_functions": ["ood_functions/**"],
}


def _build_motion_models() -> None:
    """Derive final_model/ and final_model_for_ood/ from the downloaded training run."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_motion_models  # noqa: E402  (local script, same dir)

    run_dir = MODELS_DIR / "motion_prediction" / "final_training_run"
    if not (run_dir / "checkpoints").is_dir():
        print(
            f"  [skip build] {run_dir}/checkpoints not present "
            "(download the motion_prediction group to build the deployable models).",
            file=sys.stderr,
        )
        return
    build_motion_models.build(str(run_dir), str(MODELS_DIR / "motion_prediction"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Source HF repo id (namespace/name).")
    parser.add_argument(
        "--only",
        choices=sorted(DOWNLOAD_GROUPS),
        help="Download only one group (default: all groups).",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not build final_model/ and final_model_for_ood/ after downloading.",
    )
    parser.add_argument("--token", default=None, help="HF token (defaults to cached login / HF_TOKEN).")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        print(
            "huggingface_hub is not installed. Run `pip install -e .` (it is now a dependency) "
            "or `pip install huggingface_hub`.",
            file=sys.stderr,
        )
        return 1

    groups = [args.only] if args.only else list(DOWNLOAD_GROUPS)
    allow = [pat for g in groups for pat in DOWNLOAD_GROUPS[g]]
    print(f"Downloading from https://huggingface.co/{args.repo_id} -> {MODELS_DIR}/")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=REPO_TYPE,
        local_dir=str(MODELS_DIR),
        allow_patterns=allow,
        token=args.token,
    )

    if not args.no_build and ("motion_prediction" in groups):
        print("Building deployable motion models (final_model/, final_model_for_ood/)...")
        _build_motion_models()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download the model checkpoints required by the pipeline into the ``models/`` folder.

This is a **placeholder**: fill in ``MODEL_MANIFEST`` below with the real download URLs
(e.g. a university server, an S3/GCS bucket, or a Hugging Face Hub repo) and, if needed,
adapt :func:`download_file` to your hosting (auth headers, ``huggingface_hub``, etc.).

Layout this script populates (relative to the repo root)::

    models/
      pose_estimation/H36M/RegressFlow/seed_420/...        # JAX RegressFlow pose nets
      motion_prediction/final_model/dct_pose_transformer.pickle
      motion_prediction/final_model_for_ood/<score_fn>.cloudpickle
      yolo/yolo11n-pose.pt, yolo26n-pose.pt, ...           # YOLO detector/pose weights

Run::

    python scripts/download_models.py            # download everything
    python scripts/download_models.py --only yolo
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# ---------------------------------------------------------------------------
# TODO: fill in real URLs. Each entry maps a logical group -> list of
# (url, destination-relative-to-models/) pairs.
# ---------------------------------------------------------------------------
MODEL_MANIFEST: dict[str, list[tuple[str, str]]] = {
    "pose_estimation": [
        # ("https://YOUR_HOST/regressflow_seed_420.zip",
        #  "pose_estimation/H36M/RegressFlow/seed_420/regressflow.zip"),
    ],
    "motion_prediction": [
        # ("https://YOUR_HOST/dct_pose_transformer.pickle",
        #  "motion_prediction/final_model/dct_pose_transformer.pickle"),
    ],
    "yolo": [
        # ("https://YOUR_HOST/yolo26n-pose.pt", "yolo/yolo26n-pose.pt"),
    ],
}


def download_file(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (override for auth / HF Hub / cloud SDKs)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url}\n    -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - trusted, user-configured URLs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        choices=sorted(MODEL_MANIFEST),
        help="Download only one group (default: all groups).",
    )
    args = parser.parse_args()

    groups = [args.only] if args.only else list(MODEL_MANIFEST)
    total = sum(len(MODEL_MANIFEST[g]) for g in groups)

    if total == 0:
        print(
            "No download URLs configured yet.\n"
            "Edit MODEL_MANIFEST in scripts/download_models.py with the real checkpoint URLs,\n"
            "then re-run this script. The expected models/ layout is documented in models/README.md.",
            file=sys.stderr,
        )
        # Still create the directory skeleton so paths exist.
        for sub in ("pose_estimation", "motion_prediction", "yolo"):
            (MODELS_DIR / sub).mkdir(parents=True, exist_ok=True)
        return 1

    for group in groups:
        print(f"[{group}]")
        for url, rel_dest in MODEL_MANIFEST[group]:
            download_file(url, MODELS_DIR / rel_dest)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Models

Model **checkpoints** (artifacts) live here. The actual weight files are git-ignored — only
this README is tracked. They are hosted on the Hugging Face Hub:

**https://huggingface.co/JakobThumm/conformal-human-motion-prediction-models**

Fetch them (and build the deployable motion models) with:

```bash
python scripts/download_models.py            # everything
python scripts/download_models.py --only pose_estimation
```

This needs `huggingface_hub` (installed by `pip install -e .`). The repo is public, so no
login is required to download.

## Expected layout

```
models/
├── pose_estimation/                         # JAX RegressFlow nets (downloaded)
│   ├── jax_resnet18_regressflow_{args.json,params.pickle}
│   ├── jax_resnet18_regressflow_3joints_{args.json,params.pickle}
│   ├── jax_resnet50_regressflow_{args.json,params.pickle}
│   └── camera-parameters.json
├── motion_prediction/
│   ├── final_training_run/                  # per-stage Orbax checkpoints + exports (downloaded)
│   ├── final_model/                         # built: full DCTPoseTransformer
│   ├── final_model_for_ood/                 # built: OOD head, random projection (DEFAULT)
│   └── final_model_for_ood_fixed_joints/    # built: OOD head, 9 hand-picked coords (option)
└── ood_functions/                           # sketched-Lanczos OOD score fns
    ├── jax_resnet18_regressflow_3joints_score_fn.cloudpickle
    ├── dct_pose_transformer_randproj_score_fn.cloudpickle       # motion OOD (default)
    └── dct_pose_transformer_fixed_joints_score_fn.cloudpickle    # motion OOD (option)
```

`final_model/` and the `final_model_for_ood*/` dirs are **not** downloaded — `download_models.py`
derives them locally from `final_training_run/` via
[`scripts/build_motion_models.py`](../scripts/build_motion_models.py), so the OOD readout heads stay
in sync with the code (`OOD_PROJECTION_DIM` / `OOD_PROJECTION_SEED`, `REDUCED_JOINT_INDICES`).

The motion score functions are head-specific and their scores are **not** comparable in magnitude.
The Hub may still host a legacy `dct_pose_transformer_score_fn.cloudpickle`; that file is the
fixed-joints head under its old name. Rebuild the score functions locally (README section 2.3) rather
than relying on the hosted copy after a head change.

The `old_models/` folders (legacy / superseded checkpoints) are intentionally **not** hosted.

## Hosting (maintainers)

To (re-)publish the checkpoints to the Hub:

```bash
hf auth login                       # or export HF_TOKEN=hf_...
python scripts/upload_models.py     # pose_estimation/ (minus old_models) + final_training_run/ + ood_functions/
```

## How these are referenced

- Pose scripts: `--model_save_path models/pose_estimation`
- Motion scripts: `--motion_model_save_path models/motion_prediction/final_model/dct_pose_transformer.pickle`
- Motion OOD score function (default head):
  `--motion_score_fn_path models/ood_functions/dct_pose_transformer_randproj_score_fn.cloudpickle`

The model **definitions** (code) live in `src/conformal_human_motion_prediction/models/`, not here.

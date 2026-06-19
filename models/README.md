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
│   ├── final_model/                         # built from final stage: full DCTPoseTransformer
│   └── final_model_for_ood/                 # built from final stage: reduced-output (OOD)
└── ood_functions/                           # cached sketched-Lanczos OOD score fns (downloaded)
    ├── jax_resnet18_regressflow_3joints_score_fn.cloudpickle
    └── dct_pose_transformer_score_fn.cloudpickle
```

`final_model/` and `final_model_for_ood/` are **not** downloaded — `download_models.py` derives
them locally from `final_training_run/` via [`scripts/build_motion_models.py`](../scripts/build_motion_models.py),
so the reduced-output OOD args stay in sync with the code (`REDUCED_JOINT_INDICES`).

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
- OOD score function: `--motion_score_fn_path models/ood_functions/dct_pose_transformer_score_fn.cloudpickle`

The model **definitions** (code) live in `src/conformal_human_motion_prediction/models/`, not here.

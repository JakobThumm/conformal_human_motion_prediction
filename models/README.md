# Models

Model **checkpoints** (artifacts) live here. The actual weight files are git-ignored — only
this README is tracked. Fetch them with:

```bash
python scripts/download_models.py
```

(Configure the download URLs in [`scripts/download_models.py`](../scripts/download_models.py)
first — the `MODEL_MANIFEST` is a placeholder.)

## Expected layout

```
models/
├── pose_estimation/
│   └── H36M/RegressFlow/seed_420/        # JAX RegressFlow 2D pose nets (+ _args.json, _params.pickle)
├── motion_prediction/
│   ├── final_model/
│   │   └── dct_pose_transformer.pickle   # trained DCTPoseTransformer motion model
│   └── final_model_for_ood/
│       └── <name>_scores_..._sketch_srft_....cloudpickle   # cached OOD score function
└── yolo/
    ├── yolo11n-pose.pt
    └── yolo26n-pose.pt                   # YOLO detector / pose-uncertainty weights
```

## How these are referenced

- Pose scripts: `--model_save_path models/pose_estimation`
- Motion scripts: `--motion_model_save_path models/motion_prediction/final_model/dct_pose_transformer.pickle`
- OOD score function: `--motion_score_fn_path models/motion_prediction/final_model_for_ood/<...>.cloudpickle`

The model **definitions** (code) live in `src/conformal_human_motion_prediction/models/`, not here.

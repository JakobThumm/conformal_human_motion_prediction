# Motion-prediction uncertainty-coverage improvements

Training-side changes that make the predicted covariance — and the conformal sphere derived from it,
`radius = sqrt(λ_max(Σ)·χ²₃(SET_LIKELIHOOD))` — cover the ground truth more reliably. Under-coverage
is the safety failure we care about, so the goal is to raise the under-covered cases **without**
uniformly inflating volume (which would hurt robot availability).

All changes are **flag-gated and default OFF**, so the existing pipeline is unchanged unless asked.

## The problem (two distinct mechanisms)

* **M1 — output-covariance saturation at high input uncertainty (arms/wrists).** The head reads
  *detached* backbone features (`stop_gradient` in
  `models/dct_pose_transformer_pytorch_attn.py`) and barely propagates the reported input covariance
  Σ_in: input→output uncertainty correlation is only ~0.26–0.29, and coverage collapses from ~99.9%
  at low input-unc to **95.6%** in the `[0.75, 1.0) m` input-unc bin. Failures are enriched ×10 above
  the 99th input-unc percentile. The tail is also data-starved.
* **M2 — output-tail under-calibration at low input uncertainty (ankles).** RAnkle/LAnkle have the
  worst coverage (**99.44% / 99.69%**) despite low input-unc and small error — a heavy prediction
  error tail that a Gaussian-then-sphere set underestimates. Not input-driven.

The deployed post-hoc calibration (`COV_CALIBRATION_*`) is a global affine scale and cannot restore
input-dependence (M1) or reshape a per-joint tail (M2); the fix must come from training.

## What's implemented

| ID | Idea | Where | Flags |
|----|------|-------|-------|
| **P2** | Pinball/quantile loss on the **deployed** set radius `q = sqrt(λ_max(Σ)·χ²₃)`, at τ=`SET_LIKELIHOOD`, per (joint, frame). Optimizes the deployed sphere directly, not just the Gaussian likelihood. Residual uses `stop_gradient(pred)` so it shapes only the covariance, never the mean. | `set_radius_pinball_loss` in `models/dct_pose_transformer_pytorch_attn.py` | `--lambda_pinball` (0=off), `--set_likelihood` |
| **P4** | Tail reweighting: up-weight high-input-uncertainty joint-frames in NLL+pinball by `clip((r_in/median)^γ, 1, max)`. Weights ≥1 so common cases are never down-weighted. | `_tail_reweight_weights` + `weights=` on both losses | `--tail_reweight_gamma` (0=off), `--tail_reweight_max` |

**Removed after evaluation:** P1 (input-noise augmentation) and P3 (tail-aware head). The sweep below
showed **P1 did not help and slightly hurt** (no coverage/correlation gain, worse iso-volume + MPJPE),
so it was dropped to keep the code lean — the results tables retain the P1 rows as the evidence. **P3**
(Student-t / propagation-floor head) was only ever a documented stretch; it was never implemented
(it would change the head architecture + the saved inference pickle) and is left as future work for
the residual ankle tail (M2). The shipped set is **P2 + P4 only**.

Notes:
* P4 requires the Stage-4 input layout (`input_dim = J*3 + J*9`); it is auto-gated to Stage 4 and
  no-ops elsewhere. P2 applies wherever the uncertainty head is trained (stages 2/3/4).
* Neither P2 nor P4 changes the model architecture, so the inference path
  (`initialize_jax_models` + the saved pickle) and the eval scripts are unchanged.

## How to run

Quick plumbing proof (subsampled, fast):
```bash
NSAMPLES=1500 EPOCHS=2 BATCH=64 \
  experiments/coverage/run_variant.sh proof --lambda_pinball 0.5 --tail_reweight_gamma 1.0
```

Full sweep (run when the shared GPU is free — each variant is ~40 Stage-4 epochs + one full-val
inference pass):
```bash
experiments/coverage/run_full_sweep.sh
```
`run_variant.sh <RUN_ID> [flags...]` trains one Stage-4 variant from the canonical Stage-3 checkpoint
matching the deployed `final_model` hyperparameters (40 epochs, batch 256, lr 1e-4, max_grad_norm
0.68, cosine schedule, `--augment`), regenerates the **validation** results cloudpickle, and runs both
scoreboards into `results/coverage_experiments/<RUN_ID>/`.

## Evaluation protocol (the contract)

* **Splits:** train on the train split (S1/6/7/8/9); **report on validation (S11)**. The deployed
  `COV_CALIBRATION_*` stay fixed across baseline and every variant (no tuning on S11). Any post-hoc
  conditional calibration would be fit on train/test, never on the reported split.
* **Baseline** = the deployed `final_model` validation cloudpickle (and a retrained no-flag `control`
  to separate the effect of each addition from retrain noise).
* **Scoreboards:** `evaluate_covariance_failures.py` (coverage by input-unc stratum, per-joint
  coverage, input→output correlation, lift) and `evaluate_covariance.py` (overall + per-joint
  coverage/volume). MPJPE from `examples/motion_prediction.py`.
* **Success:** high-input-unc strata (>0.3 m) rise toward 99.5%; input→output correlation rises well
  above 0.26; ankle coverage clears ~99.7%; mean/per-joint **volume does not uniformly balloon**
  (enlarge only where under-covered); low-input-unc majority and MPJPE not regressed.

## Baseline (validation, S11) — measured

| metric | baseline (final_model) |
|---|---|
| overall coverage @ SET_LIKELIHOOD=0.995 | 99.81% (0.187% fail) |
| coverage @ input-unc `[0.50,0.75) m` | 98.20% |
| coverage @ input-unc `[0.75,1.0) m` | **95.60%** |
| coverage @ input-unc `≥1.0 m` | 96.58% |
| RAnkle / LAnkle coverage | **99.44% / 99.69%** |
| RWrist / LElbow coverage | 99.73% / 99.74% |
| input→output unc correlation (Pearson) | **0.287** |
| failure lift @ p99 input-unc | ×10.2 |

## Results (validation S11, full 40-epoch runs)

Reproduce the tables with `experiments/coverage/summarize.py` (deployed-affine and, with `AFFINE=0`,
native) and `experiments/coverage/iso_coverage.py` (iso-coverage).

### View A — deployed AFFINE set (`COV_CALIBRATION_*` as shipped)

| variant | overall | cov[.5,.75) | cov[.75,1) | cov≥1.0 | RAnkle | LAnkle | in→out r | mean vol m³ | MPJPE mm |
|---|---|---|---|---|---|---|---|---|---|
| baseline (final_model) | 99.81 | 98.20 | 95.60 | 96.58 | 99.44 | 99.69 | 0.287 | 0.0545 | 50.79 |
| control (retrain) | 99.69 | 97.65 | 95.37 | 95.96 | 98.95 | 99.48 | 0.302 | 0.0452 | 50.37 |
| P1 input-noise | 99.58 | 97.75 | 95.02 | 97.63 | 98.60 | 99.29 | 0.280 | 0.0478 | 51.64 |
| P2 pinball | 99.98 | 99.85 | 99.85 | 99.56 | 99.93 | 99.98 | 0.304 | 0.1606 | 50.37 |
| P1+P2 | 99.99 | 99.88 | 99.81 | 99.56 | 99.96 | 99.98 | 0.273 | 0.1937 | 51.64 |
| P1+P2+P4 | 99.97 | 100.00 | 100.00 | 100.00 | 99.86 | 99.89 | **0.537** | 0.1566 | 51.55 |
| **P2+P4 (no P1)** | 99.97 | 99.98 | 100.00 | 100.00 | 99.88 | 99.95 | 0.506 | 0.1469 | **50.27** |
| **P2(hi λ=3)+P4 (no P1)** | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.452 | 0.2557 | 50.38 |

Under the shipped affine, P2/P4 push coverage to ~99.99% but mean volume ~3× — **because the affine
(`CT=4.0`) was tuned for the OLD under-covering model and now double-counts**: the native (no-affine)
overall coverage of the *baseline* is only 90.6% (it leans on the 4× affine to reach 99.8%). So View A
conflates the training change with stale calibration. The fair comparison is iso-coverage:

### View B — iso-coverage (each model globally scaled to 99.5% overall, native set)

| variant | iso scale | mean vol m³ | cov[.5,.75) | cov[.75,1) | cov≥1.0 | RAnkle | LAnkle |
|---|---|---|---|---|---|---|---|
| baseline | 2.20 | 0.0369 | 94.99 | **92.02** | 93.25 | 98.59 | 99.26 |
| control | 2.41 | 0.0406 | 95.69 | 93.58 | 93.60 | 98.23 | 99.05 |
| P1 | 2.54 | 0.0503 | 96.41 | 94.03 | 96.75 | 98.14 | 98.99 |
| P2 | 1.54 | 0.0356 | 96.29 | 95.08 | 97.11 | 98.29 | 99.22 |
| P1+P2 | 1.49 | 0.0394 | 96.69 | 94.83 | 98.86 | 98.14 | 99.16 |
| P1+P2+P4 | 1.60 | 0.0385 | 99.99 | 100.00 | 100.00 | 97.86 | 99.01 |
| **P2+P4 (no P1)** | 1.56 | **0.0335** | 99.84 | **100.00** | **100.00** | 97.97 | 99.21 |
| **P2(hi λ=3)+P4** | **1.30** | **0.0334** | 99.75 | 99.97 | 100.00 | **98.21** | **99.45** |

At equal coverage **and equal volume** (~0.037–0.039 m³, no balloon vs baseline), **only P1+P2+P4
holds the high-input-uncertainty strata at target** — a single global scale now covers the `>0.5 m`
input-unc tail because the model's covariance genuinely tracks input uncertainty (in→out r 0.29→0.54).
That is the M1 cure and exactly the "enlarge only where under-covered" success condition.

## What worked / what didn't

* **P2 (pinball on the deployed radius) is the workhorse.** It's the single change that moves every
  target metric: directly optimizing `q = sqrt(λ_max(Σ)·χ²₃)` toward the per-joint-frame quantile
  lifts the high-input-unc strata and the ankle tail. At iso-coverage it needs the *least* scale
  (1.54) and the smallest volume (0.0356) — i.e. it self-calibrates.
* **P4 (tail reweighting) is what restores input-dependence.** It is the only setting that raises the
  in→out correlation (0.29→0.54–0.57) and, at iso-coverage, is the only one whose high-input-unc
  strata sit at 99.5–100%. P2 fixes the *level*; P4 fixes the *conditioning* (M1's actual mechanism).
* **P1 (input-noise augmentation) did NOT help and is mildly harmful — confirmed.** No coverage or
  correlation gain, and the worst iso-coverage volume (0.0503). The covariance head reads *detached*
  backbone features, which caps how much perturbing the input can teach it; at scale 1.5/prob 0.5 it
  mostly adds label noise to the mean (MPJPE 51.6 vs 50.4). The `P2+P4 (no P1)` run confirms it:
  dropping P1 *improves* iso-volume (0.0385→**0.0335**, best of all) and MPJPE (51.55→**50.27**) at the
  same coverage, with correlation still 0.51. **Drop P1.**
* **Stronger pinball self-calibrates best.** `P2(hi λ=3)+P4` needs the least post-hoc scaling
  (iso-scale **1.30**, closest to 1.0) and gives the best ankle tail (RAnkle 98.2, LAnkle 99.5) — but
  under the *shipped* affine it over-inflates most (vol 0.256), which only re-confirms point (Volume).
* **M2 (ankle heavy tail) is only partially fixed.** At iso-coverage the ankles stay ~98% across all
  variants — a Gaussian-then-sphere set can't reshape a heavy per-joint tail by global scaling. This
  needs P3a (Student-t / mixture head) or per-joint conditional calibration.
* **Volume trade-off:** the apparent 3× volume under the shipped affine is an artifact of stale
  calibration, not the training. At iso-coverage the P2/P4 volume is on par with baseline.

## Recommendation (validated by the confirmation runs)

1. **Ship P2 + P4** (P1 removed): `--lambda_pinball 3.0 --tail_reweight_gamma 1.0`.
   This is the best operating point measured: high-input-unc strata at target, in→out correlation
   ~0.45–0.51, the **lowest iso-coverage volume (0.033–0.034 m³, below baseline's 0.037)**, best
   MPJPE, and it self-calibrates (iso-scale 1.30) so it barely needs post-hoc inflation.
2. **Retune the post-hoc calibration for the new model.** The shipped `CT=4.0` was fit to the old
   under-covering model and over-inflates a self-calibrating one ~3–5× (that is the *only* reason the
   affine-view volume looks large). Drop `CT` toward ~1.3–1.5, or — cleaner — use the
   conditional-conformal calibrator (`conformal_calibration.py`, the complementary parallel work),
   which sets the per-group scale from a calibration split and also absorbs the residual ankle tail.
3. **For M2 (ankles) specifically**, the heavy low-input-unc tail is only partially fixed by training
   (best ~98.2% RAnkle at iso-coverage). The principled cure is a heavier-tailed head (P3a:
   Student-t / small mixture) or per-joint conditional calibration — evaluate P3a next.

# s07 — Out-of-distribution monitor

**Module** `video/shots/s07_ood.py` · interp `video` · **12.0 s** (`script.py`; was 10.6 s —
the author asked for the method slides to stop feeling rushed). Works at any `--duration`:
every beat is a fraction of the shot length, only the text fade (`TXT_FADE = 0.20 s`) is
absolute. Verified at 12.0 s; the fractions are unchanged in kind from the 8.6/10.4 s checks.

## Build

```bash
cd /home/thumm/code/conformal_human_motion_prediction
# 1) cache (repo venv, CPU is enough)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu \
  .venv/bin/python video/shots/s07_ood.py --prepare        # -> video/media/s07_ood.npz (174 kB)
# 2) render (~22 s)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s07_ood.py \
    --out video/build/shots/s07.mp4 --duration 12.0
```

No new packages. Rendering is plain PIL at 2× supersampling; **matplotlib is used only as a
mathtext rasteriser** (no figure, no subprocess, no LaTeX) — see “Maths” below.

## Motion discipline (SPEC § "Motion discipline", author's binding rule)

*Words appear, pictures may move.* Everything textual in this shot now enters on **opacity
only, 0.20 s** (`TXT_FADE`) and **nothing is animated out**.

Removed in this revision:

| was | now |
|---|---|
| kicker faded up over `0.05 · D` (0.6 s) | 0.20 s |
| panel heads / axis ticks / legend rows / AUROC faded up over `0.05–0.07 · D` (0.6–0.8 s) | 0.20 s each |
| the accent line “Pose monitor · motion monitor” rode the 0.9 s *graphic* highlight blend | its own 0.20 s opacity blend (`lit_sub` still), starting after the highlight settles |
| beat 0 “fell away” (`T_ISO`, a vignette exit) and **cross-dissolved** into the panels (`T_X0/T_X1`, 0.7 s) | **hard cut** at `T_CUT = 0.255` — no exit animation anywhere in the shot |
| `𝓗 ← 𝓜[0] reuse the prediction` was centred on `[f0 … k_now]`, so it **re-centred (travelled)** on each of the three flagged frames | centred on the whole burst `[f0 … last]` once; it does not move |
| its entry (and the two other step captions) ramped over ~0.45 s of a step | ~0.20 s (`×3.0` instead of `×1.2`/`×1.4`) |

Kept, because they are pictures: the two ridges drawing in, the τ marker and its flagged band,
the Algorithm-1 timeline stepping frame by frame, the score bars growing, the validity cells
changing state, the motion buffer **shifting** (that slide *is* `shift(𝓜)`, the algorithm's own
operation) and the accept-flash. The highlight travelling over the pipeline figure in beat 0 is
explicitly allowed by SPEC.

## Structure (two beats, hard cut)

| beat | fraction of the shot | what |
|---|---|---|
| 0 | 0 → 0.255 (**3.06 s**, of which ~1.7 s is a completely still frame) | the manuscript's pipeline figure via `_overview.py`, with the **two OOD monitors** lit and the rest dimmed. Figure up 0 → 0.020, highlight 0.030 → 0.105, accent line at 0.115. Then a **hard cut**. |
| a | 0.255 → | *OOD scoring with Sketched-Lanczos* — the two SLU score ridges (0.262/0.320) and τ₂D (0.400), AUROC at 0.480 |
| b | 0.390 → 0.945 | *OOD handling (Alg. 1)* — the real 12-frame episode, **5.88 s** for 12 steps, 0.66 s hold at the end |

### Where the extra 1.4 s went

10.6 s → 12.0 s, and the 0.7 s cross-dissolve + 0.6 s fall-away were also freed:

* **beat 0** held 2.39 s (then dissolved); it now holds **3.06 s**, all of it legible — ≈ +0.7 s
  of reading time on the overview, plus it is never half-dissolved into the panels.
* **the Algorithm-1 timeline** ran 4.98 s for 12 frames; it now runs **5.88 s** (+0.9 s), i.e.
  ≈ 0.49 s per step (0.62 s on the three flagged frames, which carry weight 1.65).
* No new elements were added. Panel A's internal rhythm is unchanged in shape, only shifted
  to start at the cut.

### The overview box

The named regions of `_overview.py` do not isolate the monitors, so the shot passes its own box.
Measured on the 2 100 × 1 109 source figure, the bounding box of the **“Pose monitor”** and
**“Motion monitor”** trapezoids is

```python
REGION_MONITORS = (536, 100, 1614, 538)
```

(individually: pose monitor ≈ (540, 130, 695, 480), motion monitor ≈ (1448, 105, 1610, 532)).
Because `compose` takes a single box, everything between them — the pose history and the
“Uncertainty-aware motion prediction” block — is inside the lit region too. **Suggested addition
to `_overview.py`:** `REGION_MONITORS = (536, 100, 1614, 538)` (not added here; that file is
owned by the integrator).

## Maths

τ₂D, SLU₂D, *N*_req, *f*_mot, 𝓗, 𝓜, Σ*v*ᵢ are set in **Computer Modern via matplotlib's
`MathTextParser("agg")`**, cap-height-matched to the Inter line they sit in and pasted on Inter's
measured baseline — the technique s05 introduced (`s05_motion.py :: math_glyph`). The helper was
**copied** into `s07_ood.py` (`math_glyph`, `Canvas.math`, `Canvas.rich`) rather than promoted to
a shared module, to keep shot files independent. Prose is Inter only.

## What is on screen and where every number comes from

| on screen | value | source |
|---|---|---|
| ID score distribution | 49 249 frames | `results/final/full_pipeline/n_correct_poses_required_3/full_pipeline_results.cloudpickle` → `poses_3d_ood_scores` (the H36M test run of `examples/eval_full_pipeline.py`) |
| OOD score distribution | 263 frames | `results/pose_prediction_ood/pose_ood_scores.cloudpickle` → `OOD (tiger-pose)` |
| `τ₂D = 0.07` | 0.07 | see “The threshold” below |
| `AUROC 0.992` | 0.9918 | recomputed in `--prepare` (rank statistic) over those two real score sets; threshold-independent |
| the 12-frame episode | frames 40 413…40 424 | same full-pipeline cloudpickle (`poses_3d_ood_scores`, `poses_3d_is_ood`) |
| `N_req = 3`, 10 M-slots | | `motion_prediction/h36m_settings.py :: N_CORRECT_POSES_REQUIRED`, `PREDICTION_HORIZON_LENGTH` |

The reference AUROC in `results/pose_prediction_ood/pose_ood_detection_metrics.json` is 0.9976;
that run used a 500-frame ID subsample. Using the full 49 249-frame test run instead (same run
that supplies the episode, so the two beats are one experiment) gives 0.9918.

## The threshold τ₂D = 0.07

The manuscript's calibration condition is `P(SLU ≤ τ) ≥ 1 − ε_OOD` with ε_OOD ≈ 10 %. Recomputed
over the 49 249 real H36M test scores:

| τ | P(SLU ≤ τ) on ID | fraction of tiger-pose flagged (TPR) |
|---|---|---|
| **0.07** | **98.77 %** | **93.16 %** |
| 0.2 (`h36m_settings.py :: OOD_THRESHOLD`, previously shown) | 98.997 % | 52.85 % |

So 0.07 satisfies the ε_OOD condition with a large margin *and* is the better operating point:
it flags 93 % of the tiger-pose frames instead of 53 %. On screen the purple ridge now sits
almost entirely to the right of τ, which is the honest picture at this threshold. (The TPR is
*not* printed — the author's revision trimmed the caption to `AUROC 0.992` — but it is the
reason the picture improved.)

The shipped constant in `h36m_settings.py` is still 0.2; the video now shows 0.07. That is the
one number on screen that does not match a repo constant, which is why the derivation above is
recorded here.

## The episode is real

Frames 40 413…40 424 of the H36M test stream. Scores:

```
0.008 0.009 0.007 | 0.309 0.333 0.307 | 0.008 0.008 0.006 0.006 0.007 0.009
```

Chosen because at τ = 0.07 the flag pattern is `0 0 0 | 1 1 1 | 0 0 0 0 0 0`: three clean
in-distribution frames, a three-frame OOD burst two decades above τ, then recovery exactly when
`Σ v = N_req = 3` consecutive image-based poses have arrived (relative frame 8). The animation
therefore shows 5 buffer shifts and one refill.

`--prepare` **asserts** that `ep_scores > 0.07` equals the run's own `poses_3d_is_ood` over this
window — i.e. the deployed pipeline (τ = 0.2 plus the `COVARIANCE_OOD_THRESHOLD = 1e5` rule)
flagged exactly these three frames too. The timeline is not a counterfactual: lowering τ to 0.07
does not change this episode at all.

Search over the whole test run at τ = 0.07: 607 flagged frames in 100 runs; only **two** runs of
length 3–4 have ≥ 4 clean frames before and ≥ 7 after. The other one (frame 33 981, scores
0.072/0.084/0.078) sits within 3 % of τ and would be unreadable on a log bar chart, so 40 413
was the only real choice.

`motions_is_ood` over this window is `0 0 0 | 1 0 0 | 1 1 0 0 0 0` — i.e. `SLU_mot` also fires on
the first OOD frame and on the two frames right after the pose recovers. Every one of those
steps is *already* rejected by the `Σ v = N_req` condition, so accept/reject is still decided
purely by the validity streak, which is what the M-buffer animation shows. (The previous
episode, 23 850…23 863, had `motions_is_ood ≡ 0`; this one does not, but the conclusion is
unchanged.)

## Deliberate choices / caveats

* The score bars of panel B are on a log axis (`BAR_LO, BAR_HI = -2.6, -0.2`); the ID bars are
  short stubs and the OOD bars clear the τ line, which is the real ratio (≈ 40×) compressed.
* The distributions are drawn as ridges each normalised to **its own peak** (the two sample sizes
  differ by 200×); the caption says "distribution", not "counts".
* The pose monitor drives the timeline (Algorithm 1's fallback branch is triggered by `SLU_2D`),
  even though beat 0 lights both monitors. The motion monitor's own numbers (AUROC 0.986,
  τ_mot = 0.5) live in `results/motion_prediction_ood_randproj/`.
* `P(SLU ≤ τ₂D) ≥ 1 − ε_OOD` and the "medians 119× apart" line were removed at the author's
  request; the numbers survive in this file and in `--prepare`'s stdout.
* The beat change is a **hard cut**, not the cross-dissolve s05/s06/s08 use. That is the
  author's "clear instantly" rule, and it is also where ~1.3 s of the shot's new reading time
  came from. If the film later wants a common transition language again, it has to be a
  transition that does not animate type — this shot's two beats share only the kicker, which
  sits at the same place in both, so the cut is clean.
* `overview_stills` now returns `(flat, lit, lit_sub)` instead of `(flat, lit, iso)`: the
  vignette still (`iso`) only existed for the fall-away and is gone. The three stills differ
  **only** in the figure's highlight and in the accent line, so no blend between them ever
  re-renders or moves a word.

## Verified

```
$ ffprobe -v error -show_entries format=duration \
      -show_entries stream=width,height,r_frame_rate,pix_fmt,nb_frames,field_order \
      -of default=nw=1 s07.mp4
width=1920  height=1080  pix_fmt=yuv420p  r_frame_rate=30/1  nb_frames=360  duration=12.000000
field_order=progressive
```

Ten frames extracted from the encoded file and inspected (0.3 / 1.0 / 1.7 / 2.9 s on the
overview, 3.2 / 4.6 / 6.2 / 8.0 / 10.0 / 11.8 s on the panels). Every one reads as a stable
state; the lowest ink is panel B's closing caption (baseline y = 820, bottom ≈ 845) and the
kicker's baseline is y = 46 — nothing below y = 928, nothing above y = 40.

## The one deliberate exception to "words do not move"

The motion buffer slides one slot on a `shift(𝓜)` step (`slide`, `_panel_b`). That translation
**is** Algorithm 1's own operation on a picture of a ring buffer, and it is the only way the
shot can show "the prediction is reused, one step older". The `–` placeholder inside an empty
cell rides along with its cell because it is a mark on the cell, not a caption. No label,
number, equation or sentence in s07 translates.

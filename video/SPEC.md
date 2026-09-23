# Shot spec — ICRA submission video

Read this before writing any renderer. Everything here is binding; the film only holds together
because all 17 shots obey it.

## The contract

Each shot is one file `video/shots/<sid>_<name>.py`, invoked by `video/build.py` as

```bash
<PY> video/shots/<module>.py --out video/build/shots/<sid>.mp4 --duration <SECONDS>
```

(always from the repo root, `/home/thumm/code/conformal_human_motion_prediction`). It must:

1. produce **exactly** `--duration` seconds of silent H.264, **1920×1080, 30 fps, yuv420p,
   progressive**. Use `video/common.py` (`shot_args`, `FrameWriter`/`write_frames`, `conform`) —
   it guarantees the frame count and the encode settings. Never hardcode a duration: the narration
   drives it and it *will* change.
2. be **deterministic** and **re-runnable** (no interactive windows, no GUI, no network).
3. keep the bottom **14 %** of the frame (y > 928 px) free of essential content — burned-in
   subtitles live there. Same for the top 40 px.
4. import the palette/type from `video/style.py`. Do not invent colours.
5. use only **real data from this repository** for anything presented as a result. No synthetic
   stand-ins, no invented numbers. If a number appears on screen, it must be traceable to a file
   under `results/`, `datasets/`, the manuscript's result tables, or computed live by the shot.
   Cite the source in a comment.

## Interpreters

| purpose | interpreter |
|---|---|
| manim, matplotlib, cv2, numpy, piper (`interp="video"`) | `/home/thumm/miniconda3/envs/chmp-video/bin/python` |
| jax + repo models + H36M/RGB-D datasets (`interp="repo"`) | `.venv/bin/python` (repo venv) |

The repo venv needs `XLA_PYTHON_CLIENT_PREALLOCATE=false` (shared GPU). Adding a package to the
video env is fine (`/home/thumm/miniconda3/envs/chmp-video/bin/pip install ...`) — record it in
`video/shots/<sid>_notes.md` (never edit `video/README.md`, `build.py`, `script.py`, `style.py` or
`common.py`: several shots are built in parallel and those files are owned by the integrator).

**Expensive inference must be cached.** If a shot needs model inference, dataset decoding or a
long GPU job, do it in a `--prepare` step that writes a small `.npz`/`.png` into
`video/media/<sid>_*.npz`, commit-sized (< ~30 MB), and have the render path load that cache. The
render must then run in well under a minute. Put the prepare command in the module docstring and
in `video/shots/<sid>_notes.md` so the build stays reproducible.

**All source media is copied into `video/media/`** — the film must build from `video/` + the cached
artifacts alone.

## Visual language

* Background `style.BG` (#0E1117) everywhere; no white frames, no default matplotlib look.
* Semantic colour = meaning, consistently across the whole film:
  * `C_OURS` blue — our conformal prediction sets
  * `C_ISO` orange — the ISO 13855 constant-velocity baseline
  * `C_TRUTH` green — ground-truth human pose / joint
  * `C_PRED` teal — predicted mean pose
  * `C_DANGER` red — failure, contact, dangerous event
  * `C_OOD` purple — OOD monitor
  * `C_ROBOT` light grey — robot geometry
* Type: Inter (`style.FONT`). Titles ≤ 62 px, body 32 px, captions 26 px. Left-aligned labels,
  no ALL CAPS except short kickers.
* **Motion discipline (author's rule, binding).** Animation is for *graphics*, not for words:
  * **Text, numbers, equations, labels, captions, legends: entry is opacity only.** No slide-in,
    no shift, no scale, no `Transform`/`ReplacementTransform`, no travelling. Fade up over
    ≤ 0.25 s, or simply appear. A moving word costs the viewer attention that belongs to the
    content, and it eats screen time.
  * **No exit animations anywhere.** Elements stay until the shot cuts. Where a shot has several
    beats, swap between them with a hard cut (or an instant clear) — never a "move down and
    fade away".
  * Graphical elements keep their motion: 3-D scenes, camera moves, occupancy sets growing over
    the horizon, counters ticking, a highlight travelling over the pipeline figure, live footage.
    That is where animation earns its cost.
  * Ease whatever does move (`common.ease` / `common.seg`); nothing pops in linearly. Hold the
    final state for ≥ 0.6 s at the end of a shot so the cut lands on a readable frame.
* Every shot carries a small kicker label in the top-left (e.g. “2 · Motion prediction”) at
  `CAPTION_SIZE` in `FG_MUTED`, and, when it states a number, that number is on screen.
* Restraint: one idea per shot. Empty space is fine. No drop shadows, no gradients, no 3-D
  chart junk, no logos.

## Manim shots

Use the conda env's manim (0.20). Render into a **shot-private** media dir
(`config.media_dir = f"/tmp/manim_{sid}"`) so parallel builds cannot collide, then re-encode:

```python
from manim import *
import common, style
config.pixel_width, config.pixel_height = 1920, 1080
config.frame_rate = 30
config.background_color = style.BG
```

Scene length must equal `--duration` (add a final `self.wait(...)` computed from the remaining
time). Then `common.conform(rendered.mp4, args.out, args.duration)` to normalise. Manim's
`TexTemplate` works (system TeX is installed), but prefer `MathTex` sparingly — a formula on
screen should be readable in ≤ 3 s.

## Verifying your shot

```bash
cd /home/thumm/code/conformal_human_motion_prediction
<PY> video/shots/<module>.py --out /tmp/<sid>.mp4 --duration <D>
ffprobe -v error -show_entries format=duration -show_entries stream=width,height,r_frame_rate,pix_fmt -of default=nw=1 /tmp/<sid>.mp4
```

Then **look at it**: extract 4–6 frames spread over the shot and view them as images
(`ffmpeg -i /tmp/<sid>.mp4 -vf fps=1/2 /tmp/<sid>_%02d.png`). Iterate until they are beautiful,
legible at 50 % scale, and free of overlapping text. A shot that has not been visually inspected
is not done.

## Data sources (all real, all in-repo)

| what | where |
|---|---|
| H36M test predictions: `predictions/targets/covariance_matrices/ood_scores/is_oods` `(59472, 10, 13, 3)` | `results/final/conformal_prediction_sets/motion_prediction_results_test.cloudpickle` |
| calibration-set counterpart | `..._results_validation.cloudpickle` |
| conformal thresholds α (per joint × horizon step) | `models/motion_prediction/conformal_calibration*/conformal_calibrator*.npz` |
| shield simulation summary (P(F), P(W\|F), k_D, PFH_D …) | `results/final/robot_shield_risk_volume/shield_risk_volume_results.csv` |
| the 434 failing windows + the **4 dangerous trials** (window, traj, yaw, t) | `results/final/robot_shield_risk_volume/trials_ours_ood.npz` |
| robot swept capsules, 2358 trajectory phases, 4 ms grid | `datasets/robot_reachable_sets/ablation_conformal_prediction_sets_panda_4ms.csv` |
| full-pipeline / OOD-handling results | `results/final/full_pipeline/`, `results/eval_full_pipeline*/` |
| runtime table | `results/final/runtime/`, manuscript `figures/result_tables/runtime.tex` |
| real-world deployment recording (2560×1440, 30 fps, 60 s, RViz capture) | `video/media/realworld_source.mp4` |
| manuscript (numbers, wording, claims) | `Vision-Based-Safe-Human-Robot-Collaboration/content/*.tex` |

Headline numbers, already verified against the CSV — use exactly these:

* median set volume 0.090 m³ (ours) vs 0.686 m³ (ISO 13855) → **7.6×** smaller
* miss-rate **1.6 × 10⁻⁴** (ours) — 3.79 nines of reliability; ISO 5.0 × 10⁻⁶
* P(F) = 434 / 59 154 windows; P(F) upper = 1.30 × 10⁻²
* P(W|F) = **1.51 × 10⁻²** (closed form); N_W = **4 × 10⁹** trials; equivalent uniform
  placements N_D = **2.65 × 10¹¹**; 4.5 × 10⁶ trials/s on one RTX 5090
* k_D = **4** dangerous trials → **PFH_D ≤ 9.50 × 10⁻⁷ / h at 99.999 % confidence** (PL d band),
  t_cycle = 4 ms, N_h = 9 × 10⁵
* OOD handling: invalid pose buffers 16.57 % → **12.63 %** (−24 %), MPJPE 54.14 → 55.15 mm (+1.9 %)
* runtime: f_2D 13.20 ms, SLU_2D 36.39 ms, triangulation 1.87 ms, f_mot 4.69 ms,
  SLU_mot 9.08 ms, sets 0.64 ms → **65.86 ms** median total
* K_I = 50 input frames (2 s @ 25 fps), K_P = 10 predicted frames (400 ms), J = 13 joints,
  1 − ε = 99.99 %

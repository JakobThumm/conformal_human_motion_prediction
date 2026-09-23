# s05 — motion prediction (11.5 s)

Narration: *"A transformer reads two seconds of this uncertain history, and predicts the next four
hundred milliseconds, with covariances."* (6.8 s of voice; the shot is 11.5 s because it opens on
the manuscript's pipeline figure before the animation, and because the author asked for the method
slides to breathe: the extra 1.2 s went to the overview beat and to the covariance balls growing
over the horizon.)

**Motion discipline (SPEC).** All type in this shot — the kicker, the overview headline and stage
name, the two K-lines, the horizon read-out, the closing line — enters on **opacity only, 0.22 s**
(`FADE`) at a fixed position, and **nothing animates out**. Beat 1's headline/stage name are
hard-cut at the end of the cross-dissolve rather than fading with the figure. The moving things
are all graphics: the figure's highlight, the isolate + cross-dissolve, the 3-D camera orbit, the
history streaming in, the covariance balls inflating, and the horizon progress bar. The `+xxx ms`
read-out changes *value* as the horizon opens (that is data), but its label and its position do
not move.

## Files

| what | path |
|---|---|
| renderer | `video/shots/s05_motion.py` (`interp="video"`) |
| pipeline-figure helper | `video/shots/_overview.py` (shared with s03 / s04 — **not edited**) |
| shared 3-D rasteriser | `video/shots/_human3d.py` (private to s05 + s09 — **not edited**) |
| data prepare (repo venv) | `video/shots/_human3d_prepare.py` |
| cache | `video/media/s05_window.npz` (17 kB), `video/media/overview_dark.png` (built by `_overview`) |

## Build

```bash
cd /home/thumm/code/conformal_human_motion_prediction
# once — regenerates BOTH s05 and s09 caches (needs jax + spacepy + the H36M dataset)
JAX_PLATFORMS=cpu .venv/bin/python video/shots/_human3d_prepare.py
# render (~2 min)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s05_motion.py \
    --out video/build/shots/s05.mp4 --duration 11.5
```

No packages were added to the conda env (numpy + Pillow + **matplotlib** (mathtext only) + ffmpeg).

## Window: test index **13892** — H36M subject S5, action `Walking`

Chosen automatically in `_human3d_prepare.py` from the 59 472 test windows by:
`not OOD` ∧ no input-pose jump > 120 mm ∧ MPJPE in the 20–55th percentile ∧ the truth lies inside
both our conformal set and the ISO set, restricted to `Walking*`/`Greeting*` actions, then the
largest (root travel + wrist travel). Result: 1.18 m of travel over the 2 s history, 39.8 mm MPJPE
— a clean, unambiguous stride that reads in 3-D.

Index alignment between the results file and the dataset is *proved*, not assumed: the prepare step
re-instantiates the repo loader (`shuffle=False`) and asserts
`max |dataset targets − cloudpickle targets| == 0` over all 59 472 windows. (The eval DataLoader
drops the last partial batch of 16, so the results file is a 59 472-long prefix of the 59 487-window
dataset; the cache truncates accordingly.)

The window index is **no longer printed on screen** (the provenance line
"H36M · S5 · Walking · test window 13892" was removed at the author's request); it lives here.

## Beat 1 — the pipeline figure (0 → 3.48 s)

`_overview.compose(OV.REGION_MOTION, …)` draws Fig. 1 of the manuscript in the film's dark palette
with the **pose history → uncertainty-aware motion prediction** stage lit and the rest dimmed.

* `kicker=""`, `title=""`, `caption=""` — **all three strings are drawn by s05 itself**, on top of
  the *composited* frame, at exactly the coordinates/sizes `_overview` would have used
  ((64, 46) 26 px regular, (64, 92) 42 px medium, (64, 146) 30 px medium accent). Handing them to
  `compose` tied the headline's opacity to the figure's fade-up and, worse, made it fade *out*
  with the cross-dissolve. Now the headline enters on its own 0.22 s ramp at t = 0.06 s, the stage
  name at t = 0.30 s, and both are dropped in one frame at `T_X1` — a cut, not an exit animation.
* `_overview.compose` re-decodes and re-scales the cached PNG on *every* call, so s05 memoises
  `OV.dark_figure` / `OV._fit` **in its own process** (the module itself is untouched — s03/s04
  build concurrently).

## The transition (isolate → cross-dissolve, 2.22 → 3.48 s)

Not a hard cut and not a plain fade:

1. **Isolate** (`T_ISO`, 2.22 → 2.88 s). Everything outside the lit block — the other pipeline
   stages — is faded to `style.BG` through `lit_mask()`, a
   smoothstep rectangle padded by exactly its own feather (44 px) so the plateau starts on the
   block's edge: the accent outline and the panel's caption stay fully lit while the diagram falls
   away around them. The shot is left holding the single highlighted panel on a dark field.
2. **Cross-dissolve** (`T_X0 → T_X1`, 2.92 → 3.48 s). That lone panel cross-dissolves into the 3-D
   scene, whose own build has already started underneath (`T_GRID`/`T_TRAIL` begin at the first
   3-D frame and `T_NOW` at v = 0.03). The paper's little "Δt = 40/80/400 ms" skeletons therefore
   dissolve *into* the full-size walking figure rising in the same part of the frame.

Earlier attempts that were rejected on inspection: a push-in zoom on the overview (clips the
left-aligned title off the frame at any zoom centre) and a growing outline / rectangular iris
(the travelling frame reads as a UI artifact over the still-visible diagram, and the mask edge
cuts the title mid-word).

## Beat 2 — what is drawn (all of it real)

The rule is **one hero figure, everything else subordinate**. An earlier cut drew all 50 observed
poses as full skeletons and fitted the camera to the whole 1.18 m of travel; the result was a
wireframe blob in which no limb was resolvable. The current design:

* **Dotted history trail** (grey, `style.FG_MUTED`) — both wrists, both ankles and the hip
  midpoint, one dot every 2nd observed frame, alpha and dot size ramping with recency. This is
  what makes the history read as *walking*; the ankle dots trace the stride arcs.
* **7 ghost skeletons** (grey, every 6th of the last 43 observed poses) with steeply decaying
  alpha `0.035 + 0.25·age³` — a whisper of "there were more poses", nothing more.
* **The current observed pose** — one solid white skeleton, 5.4 px bones, 7.4 px joints, drawn
  with a negative depth bias so it stays readable through anything translucent in front of it.
  This is the figure the viewer locks onto.
* **Prediction** — the 10 predicted mean poses from
  `results/final/conformal_prediction_sets/motion_prediction_results_test.cloudpickle`: teal joint
  dots at all 10 steps, the in-between means as a faint fan, and the **+400 ms mean pose solid
  teal, depth-biased in front of its own spheres**.
* **Covariance balls** — `r = sqrt(chi2_3(0.9999) · λ_max(C_k^j))` from the real predicted
  covariance matrices, via the repo's own
  `utils.eval_utils.convert_covariance_matrices_to_set`. Mean radius grows **0.106 m → 0.230 m**
  across the 40 → 400 ms horizon, which is the "uncertainty grows with time" beat.

All of the input geometry is the *observed* (noisy, camera-triangulated) pose stream from
`datasets/H36M/pre_processed_motion/S5/Walking.npz` — what the model actually consumes, not mocap.

### Framing

The camera is fitted to the **hero only** (current pose ∪ predictions inflated by their spheres)
against the box `(0.400 W, 0.060 H) – (0.975 W, 0.850 H)`; the trail is then checked against a
looser box and only if it would leave the canvas does the fit fall back to the union. That puts
the skeleton at **0.58–0.61 of the frame height**, centred at x ≈ 0.69 W, with the text rail free
on the left. `YAW_OFF = 62°` off the perpendicular view both gives a 3/4 read of the stride and
foreshortens the 1.18 m of travel (to ~0.55 m on screen) so the history cannot smear the figure
across the frame. Nothing essential crosses y = 928 — the lowest element is the horizon progress
bar at y = 858; only the faint ground grid (0.2 % of the pixels in that band, ≤ #313944) reaches
into the subtitle zone, as before. The top 40 px are empty (max deviation from `BG` = 2/255).

## The text rail

Left rail, hanging indent: the legend bullet is centred at x = 70 (r = 9, so its left edge sits on
the 60 px margin) and the text hangs at x = 100.

```
◦  K_I = 50 observed poses            ← white bullet
   2 s of uncertain history @ 25 Hz
◦  K_P = 10 predicted poses           ← teal bullet
   400 ms, each joint with its covariance
   the ball grows with the horizon
   HORIZON / +xxx ms / progress bar
```

**The bullets are the legend.** The separate "● observed pose / ● predicted mean + covariance"
block was deleted; the colour of the bullet in front of each K-line now does that job (white =
the solid observed pose in the scene, teal = the predicted means and their covariance balls).

### `K_I` / `K_P` as maths

Set in **Computer Modern via matplotlib's mathtext** (`MathTextParser("agg")`), the film's single
sanctioned exception to "Inter only" (SPEC §Visual language / Type). No LaTeX subprocess and no
new dependency: matplotlib is already in the `chmp-video` env.

Both placement numbers are *measured*, never eyeballed:

* **scale** — `math_glyph()` renders `$K$` at a reference dpi, measures its cap height in the
  returned coverage raster (`baseline_row − ink_top`, with `baseline_row = height − depth`), picks
  the dpi that makes that cap ≈ 4× the Inter cap height of the line, re-measures at *that* dpi and
  downsamples by the exact measured ratio. The CM cap height therefore equals
  `Inter.getbbox("K")[3] − getbbox("K")[1]` to the pixel.
* **baseline** — mathtext's `depth` gives the pixels below the baseline, so the glyph's baseline
  row is known; it is pasted at `Inter "la"-anchor y + getbbox("K")[3] − baseline_row`, i.e. on
  Inter's own baseline. Side bearings are cropped, and the Inter run that follows starts one
  measured space (`font.getlength(" ")`) after the glyph's ink.

Rasters are cached per `(expr, size, weight)`; per frame only the tint/alpha is recomputed.

### Why the *covariance* ball and not the conformal set here

s05 sits before s06 ("a learned covariance is a guess, not a guarantee … calibrate"). Drawing the
already-calibrated conformal set in s05 would make s06's calibration a no-op on screen. So s05
shows the model's own 99.99 % Gaussian ball (`alpha = sqrt(chi2_3(0.9999)) = 4.5943`, the repo's
`conformal_calibrator_uncalibrated.npz` factor) and **s09 shows the calibrated conformal set**.
Both radii are cached (`r_model`, `r_ours`) so this is a one-line switch if the edit wants
otherwise.

## Caveat on α (applies to s09 too)

The paper's Prop. 1 writes the set as `B(p̂, α_k^j · sqrt(λ_max(C_k^j)))`. The **deployed**
calibrator `models/motion_prediction/conformal_calibration/conformal_calibrator.npz` is *not* of
that multiplicative form: `mode == "conditional"`, and
`motion_prediction.inference_helper.conformal_set_radius` applies an **additive** Mondrian/CQR
correction `r = max(r_model + q̂(joint, frame, input-uncertainty bin), 0)` over an 11-bin grid.
The only genuinely multiplicative α in the repo is the single global `alpha_max = 15.2443`
(`conformal_calibrator_max.npz`, the paper's single-threshold ablation) and the uncalibrated
`sqrt(chi2_3) = 4.5943`. The conditional calibrator is what produces the paper's headline volume
numbers (median 0.090 m³), so both shots use it and the cache also stores the *effective* per-(j,k)
α (`alpha_eff = r_ours / sqrt(λ_max)`, median 8.21, range 6.23–16.72 for this window) for
reference.

## Timeline

Beat 1 is now budgeted in **absolute seconds** (`T_OV_IN`/`T_FOCUS`/`T_ISO`/`T_X0`/`T_X1` at the
top of the module), so a re-timed narration lengthens the 3-D beat — where the covariance balls
grow — instead of stretching a diagram the viewer has already read. If the shot were ever cut
below `T_X1 / 0.42 = 8.2 s`, beat 1 is squeezed proportionally (`sq`). Beat 2 keeps its fractions
`v = (t − x0)/(D − x0)`; its *type*, however, is on the wall clock so the fades stay at 0.22 s.

| t (s) | beat |
|---|---|
| 0.00–0.22 | pipeline figure appears (opacity) |
| 0.06 / 0.30 | headline / stage name appear (0.22 s each, then static) |
| 0.24–0.74 | the rest of the pipeline dims, the motion stage lights |
| 0.74–2.22 | hold on the lit stage (1.5 s) |
| 2.22–2.88 | isolate: everything but the lit block falls away |
| 2.92–3.48 | cross-dissolve into the 3-D scene; beat-1 type is cut at 3.48 |

| v | t at 11.5 s | beat |
|---|---|---|
| 0.00–0.05 | 2.9–3.3 | ground grid |
| 0.00–0.17 | 2.9–4.4 | the dotted wrist/ankle/root trail streams in, 8 fading ghosts |
| 0.03–0.17 | 3.2–4.4 | the last observed pose solidifies **under** the dissolve |
| — | 3.68–3.90 | `K_I` line fades up (wall clock) |
| — | 5.41–5.63 | `K_P` line + horizon read-out fade up (wall clock) |
| 0.27–0.72 | 5.2–9.1 | horizon clock 40 → 400 ms; the 10 predictions + their balls inflate (3.9 s, was 2.6 s) |
| 0.72–1.00 | 9.1–11.5 | hold (2.4 s), slow orbit, "the ball grows with the horizon" |

Sphere brightness uses a "comet" weight `exp(-(hp - k - 1)/1.7)`: the frontier of the horizon glows
and older steps settle to a faint shell, so 130 spheres do not become 130 competing rings.

## Renderer

Hand-rolled painter's-algorithm rasteriser (`_human3d.py`): 2× super-sampled float canvas,
primitives (ground segments, bone capsules, joint discs, radial-gradient sphere shells) sorted by
view-space depth and alpha-composited, then box-downsampled to 1920×1080. Text is Inter drawn with
Pillow at output resolution. Camera: perspective pinhole orbiting the subject; the azimuth is
picked so the stride runs left→right in frame, +62° for a 3/4 body read, and the orbit sweeps 16°
over the 3-D beat. `fit_view()` binary-searches the orbit distance *and* a principal-point shift so
the hero geometry exactly fills its box — clear of the top 40 px and of the subtitle zone below
y = 928. `Scene.skeleton(..., depth_bias=...)` (already in `_human3d.py`, default 0.0, so s09 is
unaffected) is what lets the white and the +400 ms teal poses paint *over* translucent shells.

`_human3d.py` was **not modified** by the 10.3 s revision, nor by the 11.5 s / motion-discipline
revision (only `s05_motion.py` changed), so **s09 does not need a re-render**.

# s06 — conformal prediction sets (the centrepiece)

**Duration** 25.0 s · **interp** `video` (manim 0.20) · **file** `video/shots/s06_conformal.py`

```bash
# prepare (needs jax to unpickle the results -> repo venv; ~60 s, writes a 7 kB cache)
cd /home/thumm/code/conformal_human_motion_prediction
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu \
    .venv/bin/python video/shots/s06_conformal.py --prepare
# render (~45 s)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s06_conformal.py \
    --out video/build/shots/s06.mp4 --duration 25.0
```

Cache: `video/media/s06_scores.npz` (histogram, α_k^j, the full 10×13 α table, and the example
window's geometry). The render path never touches the 332 MB cloudpickle.

## 2026-09-22 revision: 20.0 → 25.0 s, and words stopped moving

Two changes, both from the author.

**1. +3 s on part 1, +2 s on part 2, part 3 unchanged.** `script.py` already carries
`duration = 25.0`. The extra five seconds are spent as standing time, not as longer animations —
every fade got *shorter* and the gaps between them got longer:

| block | was | now | length |
|---|---|---|---|
| beat 0 — the pipeline figure | 0.00–2.45 | 0.00–2.45 | unchanged |
| **part 1** — geometry + eq. (2) | 2.45–7.10 (4.65 s) | **2.45–10.10 (7.65 s)** | **+3.00 s** |
| **part 2** — histogram + α | 7.10–11.95 (4.85 s) | **10.10–16.95 (6.85 s)** | **+2.00 s** |
| part 3 — the conformal set | 11.95–20.00 (8.05 s) | 16.95–25.00 (8.05 s) | unchanged |

Where the +5 s actually lands: part 1 gains ~1.0 s of gap inside the geometry build (each of
the four elements now stands alone for 0.3–0.5 s before the next arrives), ~0.6 s before eq. (2)
appears, and a **1.60 s static hold on the finished equation** before the cut (it used to be
0.80 s). Part 2 gains 0.2 s of histogram growth (1.40 → 1.60 s, a graphic), a 0.6 s gap before
the quantile line and a **2.30 s static hold** on the finished histogram (was 0.45 s).
Part 3 redistributes internally: the two closing formulas land at 20.30 / 21.40 s and the shot
holds the payoff frame for **3.35 s**.

**2. `SPEC.md` "Motion discipline".** Converted or deleted:

| was | now |
|---|---|
| `FadeIn(kick)`/`FadeIn(title)` riding the 0.55 s figure fade | own 0.20 s `pl()` after the figure |
| `FadeOut(title)` during the push-in | `self.remove(title)` — hard cut |
| `FadeIn(plab/slab/tlab/dlab)` sharing 0.45–0.78 s graphic plays | own `pl()`, 0.22 s each |
| `FadeIn(form[0], shift=UP*0.12)`, `FadeIn(form[1..2], shift=UP*0.10)` | plain `FadeIn`, 0.25 s |
| `FadeOut(geo) + FadeOut(fcap) + FadeOut(form) + FadeOut(eqno)` | `self.remove(*geo, fcap, *form, eqno, ov_iso)` — hard cut |
| `FadeIn(qlab, shift=DOWN*0.12)` | plain `FadeIn`, 0.25 s |
| `FadeOut(bars/haxes/shade/qmove/rule/setline/qcap)` | `self.remove(*bars, haxes, …)` — hard cut |
| **`qlab.animate.move_to(...)` + `tag.animate.move_to(...)`** (α and its cell tag travelling from the histogram into part 3's right rail) | part 2's pair is removed at the cut and an identical `qlab2`/`tag2` is drawn, already in place |
| `FadeIn(setform, shift=UP*0.1)`, `FadeIn(guarantee, shift=UP*0.1)`, 0.50 s | plain `FadeIn`, 0.25 s |

Kept, because they are **graphics**: the three overview stills crossfading, the 1.55× push-in on
the lit block, `Create(ell)` / `Create(resid)` / `Create(rline)`, `FadeIn(pdot/tdot, scale=0.4)`,
the histogram bars growing out of the axis, the quantile marker sliding onto α
(`Transform(qmove, qline)` — the author explicitly allowed "the axis marker sliding to its
value"), and the ball inflating σ → α·σ (`Transform(sphere, big)`).

⚠ **manim gotcha found here:** the Cairo `Scene.remove` calls
`restructure_mobjects(..., extract_families=False)`, so it only drops the **exact** objects that
were added. `self.remove(geo)` where `geo`'s children were added individually is a *silent
no-op* — the first render of this revision left part 1's ellipse and equation on top of the
histogram. Hence `self.remove(*geo, …)` / `self.remove(*bars, …)`, unpacked. Objects that were
added *as* a group (`haxes`, `rule`, `setline`) are passed whole.

## Beats (25.0 s)

| t (s) | what |
|---|---|
| 0.00–2.45 | the manuscript's pipeline figure (`_overview.py`, `REGION_CONFORMAL`) fades in, kicker + title fade up, the conformal stage lights, then the rest of the page falls away (feathered isolation mask, same 44 px feather as s05) |
| 2.45–3.07 | push-in: the isolated block is scaled 1.55× **about its own centre** and faded out while the Part-1 geometry fades in; the title is removed on the same frame |
| 3.07–6.52 | one predicted position, built with a beat between each step: prediction dot + label, teal 1σ ellipse + yellow √λ_max half-axis + its mm label, green truth + label, dashed residual + its mm label |
| 7.10–8.50 | "Non-conformity score:" + eq. (2), then the same ratio in real mm → A = 2.15 |
| 8.50–10.10 | **1.60 s static hold** on the complete part 1 |
| 10.10 | **hard cut**; the calibration rule (verbatim from §"Conformal Prediction Sets") + the histogram axes |
| 10.70–12.30 | the histogram of that one cell's 39 440 scores grows out of the axis |
| 12.90–14.65 | α_k^j slides in from the right, the 99.99 % region shades, the tag + index caption land |
| 14.65–16.95 | **2.30 s static hold** on the complete part 2 |
| 16.95 | **hard cut**; the part-3 geometry + labels + α/tag in their new rail position |
| 17.75–19.65 | the sphere inflates σ → α_k^j·σ, the calibrated radius is drawn, both radii named |
| 20.30 / 21.40 | the set formula, then the guarantee |
| 21.65–25.0 | hold 3.35 s on the finished frame |

## The one cell the shot talks about

Everything — the example residual in Part 1, the histogram and α in Part 2, the two balls in
Part 3 — is the **same (joint, horizon step) cell**, so the slide tells one story:

**left knee (`JOINT_NAMES_13[9] = LKnee`), horizon step k = 3 → +120 ms.**

Source: `results/final/conformal_prediction_sets/motion_prediction_results_validation.cloudpickle`
(`predictions`, `targets`, `covariance_matrices`, shapes (39440, 10, 13, 3) / (…, 3, 3), mm;
`is_oods.sum() == 0`). Score = eq. (2): `A_k^j = ‖d_k^j‖₂ / sqrt(λ_max(C_k^j))`.

| quantity | value |
|---|---|
| n = \|Z^cal\| | **39 440** |
| split-conformal index ⌈(1−ε)(n+1)⌉, 1−ε = 0.9999 | **39 438** |
| **α_k^j (left knee, +120 ms)** | **5.4564** (2 of 39 440 windows above) |
| that cell's scores | median 0.98, p99 3.15, max 5.841 → histogram 60 bins over [0, 6], log-count axis |
| example window **1310** | ‖d‖ = **40.6 mm**, √λ_max(C) = **18.8 mm**, A = **2.153**, in-plane minor semi-axis 12.0 mm |
| α_k^j·√λ_max(C) (the drawn sphere) | **102.8 mm** |
| full α_k^j table (10×13) | min 4.514, median 8.129, max 14.619; medians by step k = 1…10: 6.36, 6.40, 7.19, 7.27, 7.96, 8.59, 8.70, 8.84, 8.95, 9.42 |
| ablation α_max (per-window maximum, *not shown*) | 14.6787 |
| analytic Gaussian √χ²₃(0.9999) (*not shown*) | 4.594 |

**Why this cell.** The author asked whether the earlier α = 14.68 was real and whether a smaller
one would visualise better. 14.68 **is** real but it is the paper's *ablation* α_max — the
quantile of the per-window **maximum** score, deliberately conservative. The α_k^j of
§"Conformal Prediction Sets" / Prop. 1 is per joint **and** per timestep, and its smallest cells
are ≈ 4.5–5.5. Picking one of those (5.46) means the calibrated sphere is only 5.5× the 1σ
ellipse, so **Part 3 needs no zoom-out at all** — the ellipse, the residual and the whole
conformal ball live in one frame at one scale (the old cut needed a 5.3× zoom). The cell is
chosen among the small-α cells at a horizon worth predicting (+120 ms, not +40 ms), and the
example window is one whose geometry reads: A ≈ 2.15 (truth clearly outside the 1σ ellipse,
comfortably inside the ball) and b/a = 0.64 (visibly an *ellipse*, not a disc).

The 2-D drawing is the real (u1, u2) plane — u1 = major eigenvector of C (sign flipped towards
the residual), u2 = the in-plane direction of the residual — rotated in plane so that **u1 is
horizontal**; the residual then leaves at its true 27.1° to the major axis. The projected
covariance is exactly `diag(λ_max, u2ᵀCu2)` (the cross term vanishes because u1 is an
eigenvector), so both ellipse semi-axes and the residual length on screen are the true ones.

## Implementation notes

* **The overview beat is inside the manim scene**, not a separate clip: `overview_frames()` calls
  `_overview.compose(REGION_CONFORMAL, …)` three times (flat / lit / lit-only) and the three
  1920×1080 arrays become `ImageMobject`s (1080 px = 8 manim units, so they fill the frame
  exactly). They carry `z_index = -5` so the kicker and the title, which are manim `Text`, ride
  on top — without that the later-added image covers them. The push-in is
  `ov_iso.animate.scale(1.55, about_point=<lit-box centre in manim units>).set_opacity(0.0)`;
  `ImageMobject.interpolate_color` blends the pixel arrays, so opacity animates fine.
  One continuous manim render → one mp4, no concatenation.
* **Mixed Inter / LaTeX lines** (`line_of`): prose via `pla.txt`, maths via `MathTex` auto-scaled
  so its cap height matches Inter at the same px (`mth`). Inter tokens are placed by their ink
  bottom — none of the strings used has a descender, so that *is* the baseline — and maths is
  centred on 0.47 × cap, which is where a formula's axis sits. A third tuple element sets the gap
  in front of a token (used to close up `⌉-th`).
* No `Text`/`MarkupText` is constructed directly anywhere; everything goes through `_pl_axis`.
  `_pl_axis.py` itself was **not** modified, so s01/s14/s17 need no re-render on its account.
* Duration is already 25.0 in `script.py`; `build.py` needs no change.

## Verification (2026-09-22 re-render at 25.0 s)

`ffprobe`: 1920×1080, 30/1, yuv420p, progressive, **750 frames, 25.000000 s**, no audio.
Frames inspected at t ≈ 0.6 / 1.8 / 2.3 / 3.4 / 4.6 / 6.6 / 7.4 / 9.8 / 10.05 / 10.5 / 12.5 /
14.8 / 16.8 / 17.0 / 17.6 / 19.8 / 20.6 / 22.0 / 24.9 s, including the frames either side of
both hard cuts (10.05 → 10.5 and 16.8 → 17.0), which are clean. With entry-only opacity the
frames read as a sequence of stable states.

Safe area checked over **all 750 decoded frames**: max luminance above y = 40 px is 16/255 and
below y = 928 px is 16/255, zero pixels > 90/255 in either band. Drawn-ink extent over the whole
shot: **rows 57 … 904** — the lowest elements are the histogram's x-axis title (descender of
"non-conformity", ~904 px) and Part 3's `α_k^j √λ_max(C) = 102.8 mm` label at ~858 px, both
clear of the 928 px subtitle band.

## Caveats / judgement calls

* The author asked to delete the "RElbow · step 8 of 10" caption from Part 1 — done. But α_k^j is
  meaningless without saying *which* joint and *which* horizon step it belongs to, so a compact
  tag **"left knee · +120 ms"** was kept next to α in Part 2 and carried into Part 3's right rail.
  Veto it by deleting `tag` in beat D if you disagree.
* Part 3's object labels are shortened to "prediction p̂" / "ground truth p" (the author's own
  words in the request) because the full "predicted position / true position" of Part 1 does not
  fit inside the ball; Part 1 has already established the full names.
* The two radius annotations in Part 3 are **colour-keyed** (yellow label ↔ yellow σ line, blue
  label ↔ blue α·σ radius) rather than joined by leader lines — there is no room for leaders
  inside the ball, and Part 1 trains the yellow = √λ_max convention 10 s earlier.
* The histogram is log-count; it now says so in the y-axis title ("calibration windows (log
  count)"). On a linear axis the tail that actually sets α would be invisible.
* Narration still says "its 99.99th percentile scales every sphere" — true of α_k^j as well; the
  only thing that changed is that the percentile is now taken per joint and per timestep, which
  is what the paper actually does (α_max is the ablation).
* **The narration no longer covers the whole shot.** `build/audio/s06.wav` is 13.93 s long and
  starts 0.15 s in, so it ends at ≈ 14.1 s. At 20.0 s the last spoken clause ("…scales every
  sphere into a conformal prediction set") landed roughly on part 3; at 25.0 s it lands in the
  middle of part 2 and **part 3 (16.95–25.00 s) plays silent**. That is a direct consequence of
  "+3 s on part 1, +2 s on part 2, part 3 unchanged" and it reads fine — the sentence names the
  thing the viewer is about to be shown — but if the silence is unwanted the fix is a longer
  narration line in `script.py`, not a shorter part 3.

# s12 — the two geometric filters (planned 9.2 s; the build stretches it to 9.324 s)

Narration: *"Almost none of those placements can reach the robot; two geometric filters prove
it. In total we check two hundred sixty-five billion placements on the GPU."*

## Revision (author's notes on the cut)

> In the video and in the paper, we simplify the PFH_D simulation for ease of understanding.
> Hence, we leave out the part with closed-form sampling from the witness region and just say we
> use filtering.

So the shot is now **the animation plus three live counts**, and nothing else:

| before | after |
|---|---|
| kicker `Results · dangerous-failure simulation` | `Results · PFH_D simulation` (maths subscript), drawn by `S11.draw_kicker` — shared with s11 |
| `7  pass level 4 · bounding spheres` | `7  pass first filter` / `entire motion sphere` |
| `2  pass level 5 · link × body` | `2  pass second filter` / `link and body spheres` |
| ball caption `witness region  W` | `First filter` |
| sub-caption `robot swept sphere ⊕ human occupancy sphere` | `robot motion sphere ⊕ human motion sphere` |
| `level 5 proves 74.5 % contact-free` | **removed** (it is a share of the *witness-region* trials; meaningless without the framing we just dropped) |
| the whole right-hand column: `P(W|F) = V(W)/V(F)`, `1.51 × 10⁻²`, "closed form — no Monte-Carlo error", "trials drawn from W", the ticking `4,000,000,000`, `= 2.65 × 10¹¹ uniform placements`, `4.5 × 10⁶ trials / s · one RTX 5090` | **removed** |

**One statistic was kept on the right**, on s11's own right-column axis (x = 1392):

```
2.65 × 10¹¹
placements checked in total
on one RTX 5090
```

It is the number the narration actually says ("two hundred sixty-five billion placements on the
GPU"), it is the same `N_D` s11's counter used as its denominator, and SPEC requires a spoken
number to be on screen. Everything else in that column was witness-region machinery and is gone.
It fades in at 0.42–0.52 D, just ahead of the words (the clause runs ≈ 3.9 → 6.0 s of 9.2 s).

Because the column is now three short lines instead of a full panel, the push-in's content shift
was relaxed from `shift_x = -170 px` to **`-60 px`**: the cell sits nearer the frame centre and
the ladder, the ball and the statistic each get their own air. The rungs became two-line
(`104 px` pitch instead of `86 px`) to keep the left column under ~440 px wide.

There is **no witness-region language on screen anywhere**. The module docstring and the notes
below still name the shield's levels 4/5 and record the closed-form `P(W|F)` — that is the
code↔simulator mapping a maintainer needs, and none of it is rendered.

## Motion discipline (SPEC § "Motion discipline", author's binding rule)

*Words appear, pictures may move.* All type enters on **opacity only, 0.20 s**
(`TXT_FADE = S11.TXT_FADE`, via the local `app(t, t0)`) at a fixed screen position, and
nothing is animated out.

| element | was | now |
|---|---|---|
| kicker | already instant (a=1.0, continuity with s11's held state) | unchanged |
| `Almost no placement can reach the robot` | 0.8 s fade | 0.20 s |
| the ladder `434` / `7` / `2` and their two-line captions | 0.75 s fade per rung | 0.20 s per rung |
| ball caption `First filter` + `robot motion sphere ⊕ human motion sphere` | faded **out** again at 0.68–0.78 D | **entry only**; it holds to the cut |
| the `2.65 × 10¹¹` block | 0.92 s fade | 0.20 s |

Kept, because they are pictures: the push-in (37 m → 9.2 m), the placement cloud evaporating
as the first filter culls it, the 5-of-7 shrinking at the second filter, the robot's swept
bounding sphere growing into the first-filter ball, and the caption's **leader**, whose far end
follows the ball through the push-in (the type at the near end does not move).

## Build

```bash
# once (repo venv) — shared with s11 (and s13, which is cut but still on disk)
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s1x_shield_data.py

# render (conda env chmp-video, ~2 min)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s12_sim_prune.py \
    --out video/build/shots/s12.mp4 --duration 9.2
```

`--stills 40,120,190,260` writes frames to `/tmp/s12_<frame>.png` (iteration only).

## Continuity

s12 opens on exactly the state s11 ends in: it imports `s11_sim_setup.build_placements` (seed 11),
so the same 434 placements are in the same spots, the camera continues s11's final
`dist / elevation / azimuth / shift`, and the chapter label is literally the same function call.
**s13 (the dangerous-trial replay) is cut from the film**, so s12 no longer has to hand anything
downstream; it ends on its own held state.

## What is computed live (not illustrated)

For each of the 434 shown placements a uniform trajectory phase `j` is drawn, exactly as the sim
draws it, and the shield's own tests are evaluated with the cached arrays:

* **first filter** (the shield's level 4):
  `|| h_c[m, k_j] − A_j || ≤ R_j + h_r[m, k_j]`, the human's entire-motion bounding sphere
  against the robot's swept-motion bounding sphere. In the robot's frame that is a single ball
  of radius `ρ = R_j + h_r` centred on `A_j` — the Minkowski sum drawn on screen (for the hero
  placement's own `(m, j)`). **7 of 434 pass.**
* **second filter** (the shield's level 5) — `TrialEvaluator.gate5`, per robot-link sphere ×
  per human-body sphere (`link_sphere_c/r` × `oa_true_c/r`). **2 of the 7 survive.**

That the little 434-sample demo lands on 7 and 2 is luck of the draw (seeds 11/12), but it is a
real draw from the real distribution, not a fit. For the record — not on screen any more — the
run's own closed-form `P(W|F) = 1.51 × 10⁻²` (1.6 % here) and its level-5 resolution rate is
74.5 %, `(4e9 − 1 020 761 209)/4e9` from `shield_risk_volume_results.csv`.

## Numbers on screen

| shown | source |
|---|---|
| `434` | `k_F`, `trials_ours_ood.npz: window_idx.size` — also `k_F` of the CSV row |
| `7`, `2` | computed live by `cull()` from the cached shield arrays |
| `2.65 × 10¹¹` | `equivalent_uniform_placements` of the paper run's CSV row, cached into the npz as `csv_keys`/`csv_vals`; formatted by `S11.sci`, never hardcoded |
| `one RTX 5090` | the run's hardware (`trial_rate_per_s` was measured on it) |

## Beats

| t (fraction of D) | |
|---|---|
| 0.04–0.26 | the placements outside the first filter **evaporate** (radius → 0), nearest-to-the-robot first, so the camera can fly in without pushing through them |
| 0.16–0.58 | push-in 37 m → 9.2 m, content shifted left by 60 px |
| 0.26–0.44 | the robot's swept bounding sphere appears and grows into the first-filter ball (yellow glass) |
| 0.34 (+0.20 s) | the ball's caption (`First filter`) appears with its leader — **and stays** |
| 0.40–0.52 | the second filter dims and shrinks 5 of the 7 |
| 0.42 (+0.20 s) | the `2.65 × 10¹¹` block |
| 0.52–1.00 | hold |

## Notes / caveats

* Translucent volumes ("glass") are rendered by `render_glass`, which rasterises the union at 1/2
  resolution and upsamples: a 1.5 m ball seen from 9 m covers a quarter of the frame and is the
  one genuinely expensive thing in the renderer, while being the least in need of resolution.
* Evaporating rather than merely dimming the culled placements is a presentation choice; they are
  not deleted from any statistic (the ladder keeps showing 434 → 7 → 2).
* `Overlay.tabular` (the fixed-digit-advance counter helper in `s1x_render3d.py`) is no longer
  used here — the ticking 4 000 000 000 was the only caller in this shot — but it is **kept**:
  s11's `Evaluating X / …` counter uses it.
* `≡` and `∧` are missing from Inter — use `=` and words. `⊕`, `×`, `³`, `¹¹`, `⁻²` are present.
* Maths (`PFH_D` in the kicker) is Computer Modern through `s1x_render3d.math_glyph`, cap-height
  matched to the Inter line; see `s11_notes.md` § Typography.
* No new packages in the video env.

## Verified

```
$ ffprobe -v error -show_entries format=duration \
      -show_entries stream=width,height,r_frame_rate,pix_fmt,nb_frames,field_order \
      -of default=nw=1 s12.mp4
width=1920  height=1080  pix_fmt=yuv420p  r_frame_rate=30/1  nb_frames=280  duration=9.333333
field_order=progressive
```

(rendered at the build's 9.324 s; 280 frames is the exact-frame-count contract rounding, and
`build.py` accepts < 0.08 s of drift — s09 behaves the same way.)

Seven frames extracted from the encoded file and inspected (0.3 / 1.5 / 3.0 / 4.2 / 5.5 / 7.0 /
9.2 s). Every one reads as a stable state. No overlapping text, the lowest text baseline is the
ball's sub-caption at y = 726 (nothing below y = 928, nothing above y = 40), and the last state
holds for > 0.6 s with the `First filter` caption still on screen.

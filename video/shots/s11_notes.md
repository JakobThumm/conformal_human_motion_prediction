# s11 — the PFH_D simulation, set up (12.5 s)

Narration: *"Smaller is only worth having if it stays safe. So we take every prediction failure
and drop it around the robot: random place in a ten-meter circle, random orientation, random
phase of a pick-and-place trajectory."*

## Revision (author's notes on the cut)

1. **Chapter label** — `Results · dangerous-failure simulation` → **`Results · PFH_D simulation`**,
   with `PFH_D` set as maths (roman PFH, roman subscript D, as the manuscript's
   `\mathrm{PFH_D}`). This kicker is shared with s12; both draw it through
   `s11_sim_setup.draw_kicker`, so it can only change in one place.
2. **Counter** — `X / 434 windows placed` → **`Evaluating X / 2.65 × 10¹¹ random placements`**,
   X still rising to 434 with the visible cloud. 434 placements is all that can be drawn; the
   denominator is the simulation's actual trial budget
   `N_D = equivalent_uniform_placements = 2.6455 × 10¹¹`, read from the CSV row cached in
   `video/media/s1x_shield_sim.npz` (`sci()` formats it — never hardcoded, never a literal
   exponent). X is drawn with `Overlay.tabular` inside a fixed 3-digit field, right-aligned, so
   the tail does not shuffle as the counter goes 1 → 12 → 434.

## Motion discipline (SPEC § "Motion discipline", author's binding rule)

*Words appear, pictures may move.* All type now enters on **opacity only, 0.20 s**
(`TXT_FADE`, via the local `app(t, t0)`), sits at a **fixed screen position**, and is never
animated out.

| element | was | now |
|---|---|---|
| kicker | 0.6 s fade | 0.20 s |
| `Every prediction failure, replayed around the robot` | 0.85 s fade | 0.20 s |
| the three right-column facts (`2358`, `10 m`, `uniform`) | 0.9 s fade each | 0.20 s each |
| counter `Evaluating X / 2.65 × 10¹¹ …` | 0.75 s fade, in at 0.21 D | 0.20 s, in at **0.235 D** so it lands on the frame the robot caption clears and just before the first placement lands at 0.24 D (it no longer sits at a fully-lit `0` for 0.4 s) |
| robot caption `Panda pick-and-place / real swept occupancy, replayed` | **welded to the robot's projection** (it travelled with the pull-back) and **faded out** over 0.15–0.22 D | pinned at (1064, 386); only the **leader** tracks the 3-D point. **Instantly cleared** at 0.235 D — a hard beat change, not a fade |
| `robot` marker label | welded to the robot's projection (≈ 16 px of drift over the hold) | pinned at (960, 410); the ring and the leader still follow the robot |
| `10 m` ring label | welded to the ring's rightmost projected vertex, so it slid as the camera orbited | pinned at (1726, 644); the leader still runs to the moving vertex |

Kept, because they are pictures: the camera pull-back and orbit, the Panda replaying its 2358
phases, the 434 placements raining into the disk (drop + landing flash), the 10 m ring
brightening (`ring_g`, a separate eased ramp from its label's `ring_a`), and **the counter's
digits ticking** — the number is data.

### Typography

The film is one sans family (Inter). `PFH_D` is the one symbol that needs real maths, so it is
set in Computer Modern via `matplotlib.mathtext` and **cap-height-matched to the Inter line it
sits in** — `s1x_render3d.math_glyph` / `Overlay.math`, the same technique s05 uses for K_I/K_P
(that helper measures the CM "K" at the dpi it actually renders at, so the glyph sits on Inter's
own baseline rather than on an eyeballed offset). `Overlay.advance` was added alongside it:
`measure` returns the ink bbox, which drops trailing spaces and would close the word gap.
`2.65 × 10¹¹` needs no second face — Inter has the superscript figures.

## Build

```bash
# once (repo venv, ~3 min; also feeds s12 and s13)
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s1x_shield_data.py

# render (conda env chmp-video, ~9 min)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s11_sim_setup.py \
    --out video/build/shots/s11.mp4 --duration 12.5
```

`--stills 20,120,300` writes single frames to `/tmp/s11_<frame>.png` instead of encoding — for
iteration only, the build never passes it.

## Files

| file | role |
|---|---|
| `video/shots/s1x_shield_data.py` | the `--prepare` step, **repo venv**; rebuilds the paper run's shield state and dumps `video/media/s1x_shield_sim.npz` (2.4 MB) |
| `video/shots/s1x_render3d.py` | the shared numpy ray-caster + overlay helpers (s11/s12/s13) |
| `video/media/s1x_shield_sim.npz` | the only input the render path touches |

## Rendering approach

Hand-rolled **numpy ray-caster**, no GPU, no GL, no DISPLAY (`s1x_render3d.py`):

* the only primitives in this film are **spheres** (human joint occupancies, bounding spheres) and
  **capsules** (robot links), so analytic ray/primitive intersection is exact and cheap;
* each primitive emits `(pixel, depth, primitive id)` for the samples it covers; one
  descending-depth scatter resolves the whole frame and only the surviving ~1 sample per pixel is
  shaded, so 5 650 spheres cost about what their screen area costs (~0.6 s/frame at 2x
  supersampling);
* ground = ray/plane with an analytically anti-aliased grid, a radial "pool of light" and a
  top-down soft-shadow texture; distance fog on everything;
* text/leaders are PIL over the finished frame (Inter, `style.py` palette).

mujoco/robosuite was considered and dropped: offscreen EGL is an extra failure mode for geometry
that is literally capsules and spheres.

## Data (all real)

* **Robot** — `datasets/robot_reachable_sets/ablation_conformal_prediction_sets_panda_4ms.csv`,
  the interval-0 (= actual) capsule row of each of the **2358** monitored timesteps on the 4 ms
  planning grid. The shot plays all 2358 phases across its length (9.43 s of robot motion in
  12.5 s, i.e. 0.75x).
* **Humans** — the true occupancy spheres (13 joints, DIN 33402-2 radii from
  `h36m_settings.HUMAN_RADIUS`) of the **434** prediction-failure windows of
  `results/final/robot_shield_risk_volume/trials_ours_ood.npz` (`window_idx`), i.e. the paper run's
  `k_F = 434` (row 6 of `shield_risk_volume_results.csv`, 4e9 trials, conditional-conformal +
  OOD). Rebuilt through `simulate_robot_shield.build_shield_state` with the run's own flags, so
  the geometry is bit-for-bit the simulator's.
* **Placement law** — `simulate_robot_shield.sample_robot_poses`: `(x, y)` area-uniform in a disk
  of `--pose_radius = 10 m`, yaw uniform on `[-π, π]` (seed 11 here; the sim's own seed only fixes
  the run, not this illustration).

Numbers on screen: `434` (the counter's terminal value), `2.65 × 10¹¹`, `2358`, `10 m` — all of
the above, plus `equivalent_uniform_placements` of
`results/final/robot_shield_risk_volume/shield_risk_volume_results.csv`.

## Notes / caveats

* The simulator places the **robot** around the human; the shot shows the identical relative
  geometry with the robot fixed at the origin (`place_human_in_robot_frame` applies the inverse
  transform). Stated in the module docstring.
* The placement cylinder has a z-slab of ±0.2 m. Every shown placement is drawn at `t_z = 0`, the
  slab's midpoint, so all 434 stand on one floor. (Using per-placement `t_z` would put the floor
  at 434 slightly different heights in a single picture; the ±0.2 m is invisible at this scale.)
* The 434 are of course never simultaneous in the simulation — one trial is one placement. The
  crowd is a long exposure of the placement distribution, which is what the line asks for.
* Landing order is a wave sweeping from the far side of the disk toward the camera, so the
  placements nearest the lens arrive last, after the camera has pulled back.
* `script.py` lists s11 at **12.5 s** and it was re-checked there. Nothing is hardcoded — the
  beats are fractions of `--duration`.
* The counter's denominator is *not* the number of drawn placements, and the caption says
  "random placements", not "windows": each of the 434 windows is replayed at ~6 × 10⁸ placements
  in the real run. The cloud stays a long exposure of the placement law, as before.
* The change to the shared `s1x_render3d.py` is **purely additive** (`math_glyph`,
  `Overlay.math`, `Overlay.advance`); nothing existing moved, so s13 — cut from the film but
  still on disk, and carrying its own literal `KICKER` string — renders exactly as before.
* No new packages were installed in the video env (numpy / PIL / cv2 / matplotlib only;
  matplotlib is only imported for the `PFH_D` glyph, once, and the raster is memoised).

## Verified

```
$ ffprobe -v error -show_entries format=duration \
      -show_entries stream=width,height,r_frame_rate,pix_fmt,nb_frames -of default=nw=1 s11.mp4
width=1920  height=1080  pix_fmt=yuv420p  r_frame_rate=30/1  nb_frames=375  duration=12.500000
field_order=progressive
```

Eight frames extracted from the encoded file and inspected (0.5 / 1.2 / 2.8 / 4.0 / 6.0 / 7.6 /
9.5 / 12.3 s). Every one reads as a stable state: no word is caught mid-move, no caption is
mid-fade-out. No overlapping text, the bottom 14 % (y > 928) carries no essential content, the
top 40 px is clear, and the last state holds for > 0.6 s.

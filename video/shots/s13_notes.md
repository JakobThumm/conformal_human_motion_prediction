# s13 — one of the four dangerous trials, replayed exactly (11.5 s)

Narration: *"Four come back dangerous. The shield verified its trajectory, but the true occupancy
escaped the prediction and touched the robot. This is one of the four, replayed exactly."*

## Build

```bash
# once (repo venv) — shared with s11/s12
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s1x_shield_data.py

# render (conda env chmp-video, ~4 min)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s13_sim_danger.py \
    --out video/build/shots/s13.mp4 --duration 11.5
```

`--stills 45,215,343` writes frames to `/tmp/s13_<frame>.png` (iteration only).

## The trial

Row **0** of `dangerous_trials` in `results/final/robot_shield_risk_volume/trials_ours_ood.npz`:

```
window m = 238    trajectory phase j = 544    yaw = 159.6°    t = (0.361, −1.281, −0.165) m
```

### Verification (re-derived, not asserted)

`video/shots/s1x_shield_data.py` re-scores all four saved rows with the repo's own
`simulate_shield_failure_risk_volume.TrialEvaluator` on the **CPU in float64** (the paper run was
GPU/float32), after rebuilding the shield state with the run's flags:

```
trial 0: m= 238 j=  544  verified=True  contact=True  unsafe=True  level4=True  level5=True  DANGEROUS=True
trial 1: m= 238 j= 2045  verified=True  contact=True  unsafe=True  level4=True  level5=True  DANGEROUS=True
trial 2: m= 238 j=  526  verified=True  contact=True  unsafe=True  level4=True  level5=True  DANGEROUS=True
trial 3: m= 200 j= 2095  verified=True  contact=True  unsafe=True  level4=True  level5=True  DANGEROUS=True
```

**All four reproduce as verified-AND-contact**, and all four pass the level-4 and level-5 gates.
The same pass measures, with `simulate_robot_shield.segment_point_distances`:

| trial | verification clearance | contact overlap | where |
|---|---|---|---|
| **0 (replayed)** | **+2.60 mm** | **+2.96 mm** | interval 8 (t+0.16 s), link 7 × head, step 4 |
| 1 | +1.93 mm | +2.14 mm | interval 8, link 7 × head, step 4 |
| 2 | +4.27 mm | +0.35 mm | interval 7, link 7 × head, step 4 |
| 3 | +0.90 mm | +4.49 mm | interval 10, link 7 × head, step 5 |

Trial 0 was chosen because it has the largest *both* margins, so neither half of the story rests
on a sub-millimetre number.

### The window

m = 238 is a seated person leaning forward: the head drops from z = 1.16 m to 0.96 m and travels
0.35 m toward the robot over the 400 ms horizon, while the model predicts the head almost
stationary. The truth leaves the conformal set from step 4 on, peaking at
**`escape_m` = 7.3 cm** at the head (`escape_joint = 0`) at step 8 (`escape_step`) — the number
annotated on screen, computed live per step as `‖p_k − p̂_k‖ + r_true − r_set`.

## What is drawn

Display frame: `p_display = p_world − (t_x, t_y, 0)`, i.e. the robot base at the xy origin and the
floor at z = 0, keeping the placement's z-offset (the Panda's capsules start ~0.85 m above the
floor — a table-mounted cell).

| | |
|---|---|
| grey glass | the robot's **swept occupancy**: all 9 monitored intervals' planned capsules, radii grown by the `v_human = 2 m/s` bridge — exactly what `c_safe` is evaluated against |
| grey solid | the **actual** arm: the interval-0 occupancy of the 4 ms grid, advanced from phase j |
| blue glass | the **conformal prediction set** at the current horizon step (13 spheres) |
| teal | the predicted mean pose (skeleton), so the blue volume reads as a body |
| green | the **true** human occupancy + skeleton |
| red | the contacting robot link and the contacting body sphere, plus a ring marker |

Beats: verification sweep (intervals 0→8 accumulating, no intersection → `c_safe = 1`), then the
replay of wall-clock τ = 0 → 0.32 s, contact at τ = 0.16 s, then a ~2.5 s hold with the trial's
identity burned in.

**The prediction set is shown per horizon step, not as a union over steps.** That matters: the
escape is defined per step (`∃ j,k: p_k^j ∉ S_k^j`), and the step-8 true head *is* inside the
union of the step-0…10 sets. A union would have hidden the failure.

## Notes / caveats

* The monitored trajectory for phase 544 spans only τ ∈ [0, 0.16 s] (9 intervals), so the
  shield's verdict — and the recorded contact — live at τ = 0.16 s. The replay runs on to
  τ = 0.32 s (where the recorded escape peaks at 7.3 cm and the recorded arm motion has driven
  ~10 cm into the head sphere). That continuation is real recorded geometry, but it is *one*
  verification cycle played out: in operation the shield re-verifies every 4 ms. Nothing on
  screen claims otherwise; the verdict text is tied to the τ = 0.16 s instant.
* The contact is a near-tangency (3 mm). It is made legible by turning both contacting spheres
  red and stepping the blue set back to 55 % alpha once contact occurs — no geometry is moved or
  exaggerated.
* The human occupancy is drawn as **per-joint spheres**, not as the inter-joint capsules of
  manuscript eq. `O_b(t) = K(p^j, p^j', max(r^j, r^j') + r_b)`. That is deliberate: the shield
  simulation itself (`simulate_robot_shield.build_human_arrays` / `compute_human_occupancies`,
  and the `_seg_hit` kernel) tests robot capsule *segments* against human *spheres*, so spheres
  are the geometry that actually produced k_D = 4. Drawing capsules would look tidier and be a
  different object from the one being measured.
* Same renderer, palette and camera language as s11/s12 (`s1x_render3d.py`); the three shots run
  one continuous push-in from 37 m to ~5 m.
* No new packages in the video env.

## Verified

```
$ ffprobe -v error -show_entries format=duration \
      -show_entries stream=width,height,r_frame_rate,pix_fmt,nb_frames -of default=nw=1 s13.mp4
width=1920  height=1080  pix_fmt=yuv420p  r_frame_rate=30/1  nb_frames=345  duration=11.500000
```

Frames extracted from the encoded file and inspected across the shot; no overlapping text, the
bottom 14 % (y > 928) carries no essential content, and the last state holds for > 0.6 s.

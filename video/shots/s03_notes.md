# s03 — overview → 2D pose + per-joint covariance (7.3 s, `interp="repo"`)

Narration: *"It starts with two views. An adapted YOLO twenty-six returns, for every joint, not a
pixel, but a covariance."* (6.3 s of voice — there is almost no slack left.)

Two beats:

1. **0.0–2.86 s — the pipeline figure.** The manuscript's Fig. 1 (`_overview.py`, shared with
   s04/s05) appears whole (0.22 s, opacity), then the **2-D pose estimation + uncertainty-aware
   triangulation** stage lights up (0.26–0.68 s), is held legible for **1.2 s**, and the isolate +
   grow-into-footage transition runs 1.88 → 2.86 s. This beat is the pair's establishing shot —
   s04 does not repeat it.
2. **2.18 s → end — the live H36M two-view footage** (5.1 s), which is the substance of the shot.

**The 7.3 s re-cut (was 10.3 s).** Three seconds came out of the *overview* beat and out of the
live beat's tail, not out of the live beat's content: the figure beat went 3.6 s → 2.86 s (its
hold 2.5 s → 1.2 s) and the footage went 7.7 s → 5.1 s, which still leaves ≈ 2.1 s of hold on the
finished frame after the magnifier lands. To make that survive, **beat 1 is now budgeted in
absolute seconds** (`OV_REVEAL`, `OV_LIGHT`, `OV_ISO`, `OV_MOVE`, `OV_XF`, `LIVE_T0` at the top of
the module) and the live beat takes whatever is left; as a fraction of 7.3 s the overview beat
flickered. If the shot is ever cut below `OV_END / 0.42 = 6.8 s`, beat 1 is squeezed
proportionally (`sq`) so the contract still holds.

**Motion discipline (SPEC).** Every piece of type — kicker, headline, the two camera captions, the
provenance line, the magnifier label and the σ read-out — enters on **opacity only over 0.22 s**
(`FADE`) at a fixed position, and nothing animates out; the shot ends on the live frame. The
kicker and the headline are drawn *after* the composite, so the frame moves and the words do not.
Only graphics move: the figure's highlight, the isolate + grow transition, the live footage, the
skeleton/ellipse bloom and the magnifier card's fly-in. The `σ = NN × NN px` *value* follows the
clip (that is data); its label and position are static.

## What is on screen

| element | source |
|---|---|
| pipeline figure | `Vision-Based-Safe-Human-Robot-Collaboration/figures/overview_full_2.png` via `_overview.compose(OV.REGION_POSE, …)` (luminance-inverted into the film's palette; cached at `video/media/overview_dark.png`) |
| two camera panels | real H36M frames, **S11 / "Greeting" / cameras 55011271 + 60457274, frames 1524–1607** (50 fps) |
| teal skeleton | `process_frame_2d_yolo` 13-joint output (COCO-17 → 13 via `JOINT_IDX_13_MODEL`, mirror map applied) |
| blue 2σ ellipses | the same call's per-keypoint `σx, σy` from the custom ultralytics fork's **YOLO26 `Pose26`** head, turned into the diagonal 2×2 `covariance_matrix` exactly as the deployed pipeline does |
| magnifier card | crop of the *camera 2* frame at native resolution around the **left wrist** (joint 5, the joint s04 also follows), with that joint's 2σ ellipse + its principal axes |
| `σ = NN × NN px` | live per-frame `sqrt(diag(cov))` of the magnified joint, in original-image pixels |

Nothing is synthetic; no covariance is rescaled. The clip plays in slow motion (84 source frames
over the 5.1 s live beat) and both the pose and the animation settle for the last **0.6 s**
(`fsrc = min(v / 0.88, 1)`) so the cut lands on a readable, static frame.

## The figure → footage transition

Three overlapping ramps, all in seconds (`OV_ISO`, `OV_MOVE`, `OV_XF` at the top of the module):

* **`p_iso`** (1.88–2.32 s) — `isolate()` fades everything outside the lit box towards `style.BG`
  through a Gaussian-feathered rect mask, so only the highlighted stage survives.
* **`tau`** (2.06–2.86 s) — that box is then scaled up (`OV_ZOOM = 2.05`, uniform) about its own
  centre while its centre is translated onto the centre of the two camera panels — i.e. the box
  literally grows into the panels. `OV_ZOOM` is chosen so the growing box stays clear of the
  title baseline at y = 152.
* **`xf`** (2.34–2.86 s) — cross-dissolve to the live frame, which itself scales in from
  `LIVE_Z0 = 0.92`. The figure's *Left* / *Right* thumbnails land on top of camera 1 / camera 2,
  which is what makes the dissolve read as one continuous move rather than a cut.

The kicker and the title are drawn **after** the composite, at fixed positions, so the type never
warps and never jumps: the frame moves, the words do not.

## Re-running the cache

```bash
cd /home/thumm/code/conformal_human_motion_prediction
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s03_pose2d.py --prepare
```

Needs the repo venv, a GPU and `datasets/H36M/extracted/S11/Videos/Greeting.*.mp4` +
`models/pose_estimation/camera-parameters.json` + `yolo26n-pose.pt` (repo root).
Writes `video/media/s03_pose2d.npz` (~5.5 MB): JPEG-encoded native-resolution crops of both
views (one fixed crop rect per camera, covering the whole clip's skeleton ± 2σ), plus `kp`,
`cov`, `conf`, the crop rects and the clip metadata. ~40 s.

## Rendering

```bash
.venv/bin/python video/shots/s03_pose2d.py --out video/build/shots/s03.mp4 --duration 7.3
```

Offline: only `cv2`, `numpy`, `PIL`, `video/common.py`, `video/style.py`, `video/shots/_overview.py`.
~18 s, deterministic. Beat 1 is on the wall clock and beat 2 is a fraction of what is left, so a
re-timed narration lengthens the *footage*, which is what the shot is for.

## Layout

100 px margins, 28 px gutters: panels 512 × 640 at x = 100 / 640, y = 200; magnifier card
640 × 640 at x = 1180. `PW/PH` **must stay 4:5** — `prepare()` bakes that aspect into the cached
crop rects, so changing it without re-running `--prepare` stretches the footage. Captions sit on
one baseline at y = 872 (+ the σ readout at 906), clear of the subtitle band at y = 928. The type
origin (x = 100, kicker baseline 62, title baseline 152) is identical to s04's, so the cut between
the two shots does not move the headline.

## Implementation notes

* `video/shots/_pose_data.py` is the shared `--prepare` back-end for s03 **and** s04 (it only
  selects frames and calls the repo's own `process_frame_2d_yolo`,
  `create_joint_covariance_batched` and `triangulate_points_with_covariance_batched`).
* Ellipses are rasterised into a single alpha mask per layer and composited once
  (`_stamp` + `paint`), so overlapping 2σ ellipses form a **union** instead of piling up opacity —
  without that they turn into an unreadable blob. Ellipses are drawn *under* the skeleton.
* `OV.compose()` is called per overview frame (~110 frames); it returns a **read-only** array, so
  the shot copies it (`np.array(...)`) before drawing.
* No new packages installed; Inter is loaded directly from `/usr/share/fonts/opentype/inter`.

## Caveats

* The YOLO26 σ is large relative to the body (≈ 20–50 px on a ~360 px-tall person), so the 2σ
  ellipses genuinely overlap on the torso. That is the model's real output, not a drawing choice.
* The subject faces away from both cameras in this clip; it was chosen because both views keep the
  full body, both arms are raised (maximum joint separation) and the detection is stable for all
  84 frames. Changing `START_FRAME` / `N_FRAMES` in `_pose_data.py` changes the clip for *both*
  s03 and s04 — re-run both `--prepare` steps if you touch it.
* Cross-camera covariance is zero in the deployed code path, so the 2×2 per-joint covariances are
  diagonal (the manuscript's calibration-set cross-covariance is not implemented in this repo).
* At 7.3 s the narration (6.3 s) covers almost the whole shot; the last ~1 s is a hold on the
  finished frame with the clip still creeping forward and the σ read-out ticking. The overview
  hold is down to 1.2 s — legible (verified on extracted frames at 50 % scale: the lit stage is
  fully up by 0.68 s and static until 1.88 s), but it is the floor. **Any further cut has to come
  out of the live beat, or the overview beat has to go entirely** (the establishing figure would
  then have to move to s04).

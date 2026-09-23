# s04 — uncertainty-aware 3D triangulation (9.0 s, `interp="repo"`)

Narration: *"Uncertainty-aware triangulation lifts pose and covariance into three dimensions.
Every joint arrives as an ellipsoid."*

**The 9.0 s re-cut (was 7.1 s).** The author asked for the method slides to stop feeling rushed;
all +1.9 s went into *looking at the geometry*, not into new elements:

* the orbit is **slower and longer** — the same 212° → 264° sweep now runs over 7.7 s instead of
  6.5 s (`settle = min(u/0.86, 1)`), and the finished frame is held static for **1.25 s** (was
  0.57 s) so the cut lands on a readable frame;
* the two 2σ **cones** are fully drawn by 3.4 s and **stay to the cut**, and the 13 **ellipsoids**
  bloom 2.9 → 5.8 s and then have 3.2 s of quiet orbit on screen;
* `p_fade` — the ramp that used to dissolve the rays and cones away at 55–76 % of the shot — was
  **deleted**. SPEC's motion discipline forbids exit animations, and the cones are exactly what
  the shot is about, so the rays now sit at one constant alpha (`RAY_A = 0.52`) from the moment
  they are drawn and the cones never recede.

**Motion discipline (SPEC).** All type — kicker, headline, the two `camera N` captions, the
provenance line and the `0.NN × 0.NN × 0.NN m` read-out — enters on **opacity only over 0.22 s**
(`FADE`) at a fixed position and never leaves. The read-out's *numbers* change with the clip
(that is data); the label does not move. Everything that moves is a graphic: the orbit, the rays
growing towards the camera centres, the cones, the ellipsoids blooming and the live 2-D panels.

## What is on screen

| element | source |
|---|---|
| two small camera panels (left) | the same real H36M clip as s03 — **S11 / "Greeting" / cameras 55011271 + 60457274, frames 1524–1607** — with the YOLO26 2D skeleton and 2σ ellipses |
| grey back-projection rays | exact unit directions from each 3D joint towards the two **real** H36M camera centres `C = -Rᵀt` (`models/pose_estimation/camera-parameters.json`), truncated at 0.62 m, one constant alpha |
| blue cones at the left wrist | exact back-projection of that joint's 2σ *image* ellipse (both cameras) onto the joint's depth plane, plus truncated cone generators — the two cones intersect in the ellipsoid, and they stay on screen to the cut |
| 3D skeleton (teal) | `triangulate_points_with_covariance_batched` output, mm → m |
| 3D ellipsoids (blue) | the **3×3 covariances** from the same call (linear triangulation + first-order covariance propagation), drawn at 2σ: translucent projected hull + the two principal great circles through the major axis |
| `0.NN × 0.NN × 0.NN m` | live `2·sqrt(eig(C₃ᴅ))` of the left wrist, in metres |
| floor grid | 0.5 m squares at z = 0, centred on the mean ankle position (H36M world frame, z up) — drawn, but **not labelled**: the author asked for the "floor grid 0.5 m" annotation to go |

The virtual camera orbits from azimuth 212° to 264° (elevation 10° → 17°, 4.9 m out) so the
ellipsoids' anisotropy — they are stretched along the stereo depth direction, ~0.3–1.0 m at 2σ
versus ~0.1–0.3 m across — rotates into view. Pose and orbit both settle for the last ~1.25 s.

## Timeline (fractions of `--duration`, tuned at 9.0 s; type is on the wall clock)

| u | t at 9.0 s | beat |
|---|---|---|
| 0.00–0.06 | 0.0–0.5 | the two 2-D panels and the floor grid fade up |
| — | 0.00–0.28 | kicker, captions, provenance (0.22 s opacity); headline at 0.06–0.28 |
| 0.02–0.20 | 0.2–1.8 | back-projection rays grow towards the two real camera centres |
| 0.08–0.28 | 0.7–2.5 | the 3-D skeleton draws in |
| 0.17–0.38 | 1.5–3.4 | the two 2σ cones of the left wrist — and they stay up |
| 0.32–0.64 | 2.9–5.8 | the 13 covariance ellipsoids bloom |
| 0.60 | 5.4 | the `left wrist · 2σ ellipsoid` read-out appears (`NUM_U`) |
| 0.86–1.00 | 7.7–9.0 | pose and orbit settled; 1.25 s hold on the finished frame |

## Re-running the cache

```bash
cd /home/thumm/code/conformal_human_motion_prediction
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python video/shots/s04_triangulate.py --prepare
```

Same inputs as s03 (repo venv + GPU + H36M videos + camera parameters + `yolo26n-pose.pt`).
Writes `video/media/s04_triangulate.npz` (~2.7 MB): `p3` (mm), `C3` (mm²), `cam_centers`, the two
projection matrices, the 2D keypoints/covariances and 0.75-scale JPEG crops of both views. ~40 s.

## Rendering

```bash
.venv/bin/python video/shots/s04_triangulate.py --out video/build/shots/s04.mp4 --duration 9.0
```

Offline (`cv2`, `numpy`, `PIL`, `common`, `style`), ~36 s, deterministic. All *graphical* timing is
a fraction of `--duration`, so the shot re-times with the narration; the text fades and the
read-out cue are on the wall clock (`FADE`, `NUM_U · duration`) so they stay short.

## Type

Kicker **"Perception"** (no "Method 1 ·" prefix — the film's kicker scheme is the bare chapter
name), headline **"Uncertainty-aware 3D triangulation"**. Both sit at exactly the same origin as
s03's (x = 100, baselines 62 and 152) so the cut from s03 does not move the headline. The shot
carries no overview beat of its own: s03 opens the perception pair on the pipeline figure, s04
picks the pair up from there. (Both shots play the same 84-frame clip from its start, so the cut
does rewind the footage — the two skeletons differ in pose across the cut. Unchanged behaviour,
noted here in case it ever needs fixing.)

## Implementation notes

* Shared `--prepare` back-end: `video/shots/_pose_data.py` (same module s03 uses — one clip, one
  inference pass, two caches).
* The 3D viewport is a hand-rolled look-at pinhole (`Cam`) rendered with cv2 sub-pixel AA rather
  than matplotlib-3D, for control over depth order and alpha.
* `Layer` batches every anti-aliased shape of one pass into a single 8-bit buffer (drawn in
  ascending alpha so overlaps take the max) and composites once — this is what keeps the render
  at ~30 s instead of ~5 min, and it stops overlapping ellipsoids from piling up opacity.
* `S04_AZ0` / `S04_AZ1` environment variables override the orbit azimuths — handy for re-picking
  the viewpoint quickly (`--fps 3` renders a 15-frame proof in ~2 s).
* No new packages installed; Inter is loaded from `/usr/share/fonts/opentype/inter`.

## Caveats

* 2σ triangulation ellipsoids on the upper body are genuinely large (the right elbow/wrist reach
  ~1 m along the depth axis in this clip because that arm is poorly conditioned in camera 2), so
  the upper-body ellipsoids overlap. Real data — the fill alpha and ring count were reduced
  instead of shrinking the ellipsoids.
* Cross-camera covariance is zero in the repo's deployed path, and the manuscript's "small
  constant isotropic term" is not added here — this is exactly what
  `triangulate_points_with_covariance_batched` returns.
* Changing the clip in `_pose_data.py` affects s03 too; re-run both `--prepare` steps.

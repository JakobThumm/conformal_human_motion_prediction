# s09 — ours vs ISO 13855 (planned 9.0 s; the build stretches it to 9.335 s for the narration)

Narration: *"Orange is the constant-velocity model of ISO thirteen eight fifty-five. Blue is
ours: the same confidence, in seven point six times less space."*

**s10 (the volume-distribution / miss-rate panel) is cut from the film**, so this shot is now the
only place the volume comparison is stated. The read-out was therefore re-weighted: the panel
moved up (header y = 606, last baseline y = 790) and `7.6× smaller` is set at 56 px instead of
32 px, so the number the last clause of the narration names is the largest thing in the column.

## Files

| what | path |
|---|---|
| renderer | `video/shots/s09_vs_iso.py` (`interp="video"`) |
| shared 3-D rasteriser | `video/shots/_human3d.py` (private to s05 + s09) |
| data prepare (repo venv) | `video/shots/_human3d_prepare.py` |
| cache | `video/media/s09_window.npz` (17 kB) |

## Build

```bash
cd /home/thumm/code/conformal_human_motion_prediction
# once — regenerates BOTH s05 and s09 caches (needs jax + spacepy + the H36M dataset)
JAX_PLATFORMS=cpu .venv/bin/python video/shots/_human3d_prepare.py
# render (~6 min: the 0.87 m ISO spheres are huge on the 2x canvas)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s09_vs_iso.py \
    --out video/build/shots/s09.mp4 --duration 9.0
```

No packages were added to the conda env (numpy + Pillow + ffmpeg only).

## Window: test index **13327** — H36M subject S5, action `Greeting 2`

Same screens as s05 (`not OOD`, no input jump > 120 mm, MPJPE in the 20–55th percentile, the truth
inside **both** set families, walking/greeting actions), but the pick maximises *honesty* rather
than drama: among the candidates it is the window whose per-window median volume ratio
**7.63×** sits closest to the dataset-wide headline **7.61×**. So the picture does not oversell the
number that is on screen next to it. MPJPE 33.5 mm.

## How the "SaRA using ISO 13855 baseline" is constructed

(The on-screen label is the author's wording: the baseline is SARA shield *driven by* the
ISO 13855 constant-velocity human model, not ISO 13855 on its own.)

Straight out of the repo — `utils.eval_utils.compute_sara_predictions`, called exactly as
`examples/motion_prediction.py` calls it for the paper's table:

```python
in_set_m = convert_covariance_matrices_to_set(in_cov, 0.9999) / 1000.0     # [N,J] m
iso_centres, r_iso = compute_sara_predictions(
    last_input_poses=last_pose,                       # the LAST OBSERVED pose, all K_P steps
    prediction_horizon_times=[(k+1)/25 for k in range(10)],
    v_human=V_HUMAN_ISO,                              # 2.0 m/s, h36m_settings
    measurement_uncertainty=in_set_m)                 # per-joint 99.99 % input-uncertainty radius
```

i.e. `r_ISO^j(t_k) = r_in^j + v_h,max · t_k`, centred on the last observed pose and **not moving**
— the manuscript's Sec. "Integration in SARA Shield" / Sec. IV "Conformal Prediction Evaluation".
For this window the mean radius runs **0.148 m → 0.868 m** over 40 → 400 ms, against our
**0.232 m → 0.360 m**. Note the first horizon step is the one place where ISO is *smaller* than
ours (its input-uncertainty offset is tiny and 40 ms of 2 m/s is only 80 mm); by 400 ms it is
2.4× the radius and ~14× the volume. This is faithful to the per-frame rows of
`coverage_stats_sara.csv` vs `coverage_stats_conformal_prediction_sets.csv`.

## Our sets

`motion_prediction.inference_helper.conformal_set_radius` with the **deployed** calibrator
`models/motion_prediction/conformal_calibration/conformal_calibrator.npz` on the real predicted
covariances. See the α caveat in `s05_notes.md`: that calibrator is the *conditional* (Mondrian/CQR)
one, `r = max(r_model + q̂(joint, frame, input-uncertainty bin), 0)`, **not** the multiplicative
`α_k^j · sqrt(λ_max(C))` of Prop. 1. It is used because it is the calibration that produces the
on-screen 0.090 m³ / 7.6× numbers (`--method conditional`, the paper's main row). The multiplicative
form exists in the repo only as the single-threshold ablation `alpha_max = 15.2443`
(`conformal_calibrator_max.npz`, median volume 0.416 m³) and the uncalibrated
`sqrt(chi2_3(0.9999)) = 4.5943`. The cache also carries `alpha_eff = r_ours / sqrt(λ_max)`
(median 8.39 for this window) if a strictly-Prop.-1 rendering is ever wanted.

## On-screen numbers — every one traceable

| on screen | source |
|---|---|
| `0.686 m³` | `results/final/conformal_prediction_sets/coverage_stats_sara.csv` → `overall_volume_p50_m3` |
| `0.090 m³` | `results/final/conformal_prediction_sets/coverage_stats_conformal_prediction_sets.csv` → `overall_volume_p50_m3` |
| `7.6×` | ratio of the two, computed in the prepare step (7.607) |
| `2.0 m/s` | `motion_prediction/h36m_settings.py::V_HUMAN_ISO` |
| `99.99 %` | `SET_LIKELIHOOD = 0.9999`, also the calibrator's `level` |
| `59 472 test windows` | `n_predictions` in both CSVs |

Read them as the *median per-sphere set volume over the whole H36M test split* — that is what the
caption says; the frame itself shows one window.

## Motion discipline (SPEC § "Motion discipline", author's binding rule)

*Words appear, pictures may move.* Every label, number and caption enters on **opacity only,
0.20 s** (`TXT_FADE`, expressed as `TXT_FADE / D` so the entry stays 0.20 s at any duration),
through the local helper `app(u, u0)`. Nothing in this shot is animated out.

What was removed in this revision — all of it *entry-length*, since s09 never had an exit or a
slide to start with:

| element | fade-in before | now |
|---|---|---|
| kicker `Results · how tight are the sets?` | 0.65 s (`seg(u, 0.00, 0.07)`) | 0.20 s |
| the window ID, top right | 0.93 s | 0.20 s |
| `SaRA using ISO 13855 baseline` + its two sub-lines | 0.93 s | 0.20 s |
| `Ours` + `conformal prediction sets` + `99.99 % confidence` | 0.93 s | 0.20 s |
| `ground truth · inside both` (swatch + label) | 0.93 s | 0.20 s |
| `HORIZON` / `+XXX ms` | 0.75 s | 0.20 s |
| `0.686 m³`, `0.090 m³` | 0.93 / 0.84 s | 0.20 s each |
| the rule + `7.6× smaller` | 0.84 s | 0.20 s |

Kept, because they are pictures: the camera orbit, the ground grid, the green ground-truth
trail, the orange ISO family inflating along the horizon clock, our blue sets appearing along
the same clock, and **the orange family stepping back to 14 % opacity** at `T_FADE`. That last
one is geometry retreating so the blue compactness lands — it is not text and it is not a
disappearing caption, so it stays. The `+XXX ms` horizon read-out counts up with the geometry:
that is data ticking, like s11's placement counter.

`_human3d.py` was **not** touched (s05 shares it and is being revised in parallel); nothing in
this shot needed a rasteriser change.

## Timeline (fractions of `--duration`)

Beats are pinned to the narration wav (`video/build/audio/s09.wav`, 7.77 s at `audio_start`
0.15 s): the sentence break is at 5.60 s of the shot (u = 0.62) and "… in seven point six times
less space" runs ≈ 6.7 → 7.9 s (u = 0.74 → 0.88).

| u | beat |
|---|---|
| 0.00–0.08 | grid, kicker |
| 0.05–0.17 | the last observed pose (white) + the green mocap ground-truth future |
| 0.16–0.44 | orange ISO family inflates along the horizon clock 40 → 400 ms |
| 0.26–0.36 | the volume panel opens with `0.686 m³` |
| 0.50–0.70 | blue conformal sets appear along the same clock — lands on "Blue is ours" |
| 0.58–0.67 | `0.090 m³` |
| 0.70–0.82 | orange steps back to 14 % opacity — the blue compactness lands |
| 0.71 (+0.20 s) | `7.6× smaller`, then hold + slow orbit to the cut |

(The right-hand column of the table above is now "start of a 0.20 s opacity entry", not a ramp
interval; the *graphic* intervals, 0.16–0.44 and 0.50–0.70 and 0.70–0.82, are unchanged.)

Both families use the same "comet" sphere weighting as s05 (`exp(-(hp-k-1)/τ)`): only the frontier
of the horizon glows (τ = 0.9 for ISO, 1.1 for ours), so 10 nested ISO shells do not turn into 10
competing rings. A linear taper then removes shells more than ~3 horizon steps behind the frontier
entirely: their rim alpha is already < 0.01, so it is visually imperceptible, and it cuts the
per-frame sphere count from 130 to ~26 per family. That matters here because a 0.87 m ISO sphere
covers a ~400-px-radius disc (800 px on the 2× super-sampled canvas).

Render time is ~4 min for 270 frames (the ISO spheres dominate the rasteriser); s05 is ~100 s.

## Colour discipline

`C_ISO` orange = ISO 13855 baseline, `C_OURS` blue = our conformal sets, `C_TRUTH` green = mocap
ground truth, white = the last observed pose, which is the *centre* of the ISO set.

The ground truth is drawn as **the +400 ms mocap skeleton plus a dotted 10-step joint trail**
rather than 10 stacked skeletons: this window is an in-place greeting, so ten overlaid green
skeletons collapse into a green blob. The dots make each of the ten future joint positions
individually visible inside both set families, which is the point of the colour.

## Caveats

1. The narration says "the same confidence". Strictly, on the test split the ISO set's marginal
   coverage is 99.9995 % and ours is 99.9837 % (both ≥ the 99.99 % target), so ISO is slightly
   *more* conservative as well as much larger. The shot therefore states only "99.99 % confidence"
   for ours and "7.6× smaller" for the comparison — it does not claim equal coverage on screen.
2. The framing is fitted to the ISO family, so when the orange fades the blue looks small in a
   large frame. That is deliberate: the empty space is the 7.6×.

## Verified

```
$ ffprobe -v error -show_entries format=duration \
      -show_entries stream=width,height,r_frame_rate,pix_fmt,nb_frames,field_order \
      -of default=nw=1 s09.mp4          # rendered at the build's 9.335 s
width=1920  height=1080  pix_fmt=yuv420p  r_frame_rate=30/1  nb_frames=280  duration=9.333008
field_order=progressive
```

Eight frames extracted from the encoded file and inspected (0.3 / 1.0 / 2.2 / 3.6 / 5.4 / 6.8 /
8.0 / 9.2 s). Every one reads as a stable state — no word is caught mid-fade or mid-move.
Nothing above y = 40 (the kicker's baseline is y = 44) or below y = 928 — the lowest ink is the
`7.6× smaller` line, bottom ≈ 860. The final state holds from u = 0.82 (1.7 s).

`_human3d.py` was **not** touched by this revision, so s05 did not need re-rendering.

# s08 — "The spheres become capsules … No intersection, go. Otherwise, fail-safe stop."

Renderer: `video/shots/s08_shield.py` (interp `video`).
Shared plumbing: `video/shots/_realworld_common.py`, `video/shots/_overview.py`.

```bash
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s08_shield.py \
    --out video/build/shots/s08.mp4 --duration 14.7
```

Render time ~50 s. No new packages. Cached artefacts:
`video/media/realworld_shield_state.npz` (regenerate with
`… video/shots/_realworld_common.py`) and `video/media/overview_dark.png`
(written on first use by `_overview.dark_figure()`).

The shot now **plays before s07 (the OOD monitor)** — it is the last method step
before the results — and runs **14.7 s** (was 9.7).

## Beat 0 — the overview figure (0.0 – 3.0 s)

`_overview.compose()` with the **safety-verification** region lit in `C_OURS`,
kicker *Safety verification*, title *Verify the monitored trajectory* (both drawn
by the shot, not by `compose`, so they do not jump position when the footage
takes over — same pattern as s03). Held ~2 s, then the rest of the pipeline
dissolves away (2.05–2.55 s), the lit block alone grows ×1.32 toward the live
action and cross-dissolves into the 3-D view (2.40–3.00 s). Same transition
language as s03 / s05 / s06. The title no longer fades out into the transition:
it fades up once and **stays for the whole shot** (see "Motion discipline").

**`OV.REGION_SHIELD` is now the entire bottom row.** The author asked for the
highlight to cover the whole bottom row of the figure *including SARA shield*,
and the shared constant was widened for this revision to
`REGION_SHIELD = (2, 548, 2098, 1107)` — intended trajectory → monitored-
trajectory planner → robot reachable occupancy → **Verification** (eq. (7),
`O^r_i([t0,tM]) ∩ O^h_b([t0,tM]) = ∅`) → the reachable occupancies built from the
conformal prediction sets. That row *is* SARA shield; lighting only its right
half told half the story. **The shot's local override
`REGION = (1000, 555, 2098, 1105)` is deleted** — s08 now simply uses
`OV.REGION_SHIELD`.

**The grow had to be retuned.** On canvas the lit block is now 1310 × 358 px
centred at (960, 722) — two thirds of the frame width — instead of the
692 × 353 of the old half-row. The previous `OV_ZOOM = 1.55` blew it up to
1310 × 1.55 = 2030 px wide, i.e. past *both* frame edges: the ends of the row (and of eq. (7))
fell off screen before the dissolve finished. The shot now uses
**`OV_ZOOM = 1.32`** (→ 1728 × 473, still inside 1920) and
**`OV_TARGET = (930, 430)`**, the centre of the live geometry. Rendered and
inspected at 2.2 / 2.6 / 2.9 s: the whole row stays in frame, keeps growing
toward the 3-D scene, and the cross-dissolve lands on the live capsules exactly
as before.

## Beat 1 — live shield output (2.45 – 14.7 s)

Source `video/media/realworld_source.mp4`, the RViz 3-D view — **live SARA
shield output**, not a re-render.

The author asked for *more verified motion*. The recording has no single
contiguous stretch that is both a long verified run **and** ends in a clean
intersection (see the shield-state runs below), so the shot uses **two moments
of the same continuous capture joined by a 0.4 s cross-dissolve**. The two
moments frame almost identically (capsules left, occupancy right, same camera),
so the dissolve reads as a time jump rather than a cut.

| out | source | speed | why |
|---|---|---|---|
| 2.45 – 6.88 s | 46.200 → 50.100 s | **0.88×** | **clip A** — 3.9 s of verified operator motion, the sets closing in with a clear gap |
| 6.88 – 7.28 s | cross-dissolve | — | into clip B (extrapolated out of clip A, `Timeline` xfade) |
| 6.88 – 8.73 s | 33.300 → 34.133 s | **0.45×** | **clip B** — the two sets closing to contact |
| 8.73 – 9.43 s | freeze at 34.133 s | — | last frame whose capsules are still green |
| 9.43 – 9.80 s | 34.133 → 34.500 s | 1.00× | the sets meet, capsules flip to red — untouched |
| 9.80 – 12.18 s | 34.500 → 35.500 s | **0.42×** | braking, slowed |
| 12.18 – 13.78 s | 35.500 → 37.100 s | 1.00× | braked, real time |
| 13.78 – 14.70 s | held frame | — | final read |

Playback speed is changed (0.42×–0.88× ramps plus one freeze); intermediate
frames are linearly blended (`Source.at_blend`) so the slow motion has no
judder; blending is switched off across a shield state change so a red and a
green frame are never mixed. The ordering of events and the shield state are
untouched.

Frame 1024 (34.1333 s) is the last frame with green capsules; 1025 is the first
red one. The freeze lands exactly on that boundary.

### Why this window

`display_state()` runs, in seconds of the capture:

```
VER 11.27–17.90 | BRK 17.90–19.47 | … | BRK 21.47–28.47 | VER 28.47–29.17
BRK 29.17–33.27 | VER 33.27–34.17 | BRK 34.17–46.10 | VER 46.10–50.67
BRK 50.67–53.67 | VER 53.67–57.83 | BRK 57.83–60.07
```

* 33.27–37.10 is the **cleanest** disjoint→intersecting transition: only 9
  frames (0.3 s) of boundary strobe, then ~350 frames of solid red, and the blue
  set visibly engulfs the capsules. Its verified lead-in is only 0.9 s.
* 46.10–50.67 is the longest verified run that is *usable* (11.27–17.90 is longer
  but overlaps the forbidden 11.5–14.5 s camera dolly, and its blue set drops to
  y ≈ 1000 px, over the subtitle band).
* 50.67 and 53.67→57.83→60.07 were rejected as climaxes: 50.67 strobes for 1.5 s
  and the sets never visibly overlap; 57.83 strobes for 1.1 s.

So clip A comes from 46.10–50.67 and clip B from the 34.17 event.

## Crop

3-D render area of the capture is `x 692…2542, y 98…1414`. This shot uses
`VIEW = (860, 412, 2420, 1289)` → 1560 × 877, upscaled **×1.231 with Lanczos**
to 1920 × 1080. Wider than the 9.7 s cut (`(1180, 470, 2480, 1201)`): with the
extra clip in the shot the union of the coloured geometry over *every* frame
used spanned x 0…1072, y 0…817 in the old crop, which left nowhere for the two
(now much longer) occupancy captions. In the new crop the same union is
**x 418…1287, y 45…752**, so the whole left column, the bottom band and the
right column are permanently free. Nothing outside the rect (options tree, image
panels, toolbar, title bar, cursor, the "31 fps" read-out) is ever on screen.

## Overlays (all drawn by this shot, in `video/style.py` colours)

* kicker `Safety verification` top-left, title `Verify the monitored trajectory`
  — both fade up once and stay to the end of the shot;
* top-right state badge, `VERIFIED` / `FAILSAFE STOP`;
* **Human reachable occupancy** (right column, `C_OURS`), caption
  *Conformal prediction sets at the point / in time when the robot is stopped.*
  — the author's wording; it is `O^h_b(t_b)`, the occupancy operator applied to
  the conformal sets at the time the robot reaches a full stop
  (`content/S3_methodology.tex` §"Integration in SARA Shield");
* **Robot reachable occupancy** (bottom-left), caption
  *Robot occupancy along its monitored trajectory. / The monitored trajectory
  ends with a stopped robot.* — the author's wording; §"Safe Human-Robot
  Collaboration": the monitored trajectory appends a failsafe trajectory that
  brings the robot to a complete stop over `[t_0, t_b]`;
* both captions sit on a soft rounded scrim (`scrim_mask`, Gaussian σ = 30,
  62 % toward `style.BG`) because the RViz floor grid otherwise runs straight
  through the headlines and reads as a strike-through;
* bottom-right **Verification** schematic: robot disc **left** (state colour),
  human disc **right** (`C_OURS`), plus
  `Sets disjoint → verified safe` / `Sets intersect → failsafe stop`.

Both the badge and the schematic are driven **frame by frame from the measured
capsule colour**, never from a script.

### Schematic geometry

Equal radii `r = 52 px`. Centre distance `d`; the two discs interpenetrate by
exactly `2r − d`.

* disjoint: `d = 2.55 r` (gap `0.55 r`);
* intersecting: **`d = 1.60 r` → overlap `0.40 r` = 20 % of the diameter**, and
  the lens is filled in `C_DANGER` (exact scanline fill, `_lens`).

The brief also suggested `d = 1.8 r`; that is only `0.2 r` = **10 %** of the
diameter and, rendered, reads as two circles *touching*, not intersecting (it
was tried first and rejected on the acceptance criterion "clearly reads as
intersecting at 50 % scale"). `1.6 r` is the value that satisfies the brief's
prose ("overlap of 20 % of the diameter").

## Red / green semantics — evidence

`_realworld_common.compute_shield_state()` counts strongly-red vs strongly-green
pixels inside the 3-D view for all 1802 frames. The two are mutually exclusive
(≈600 px of one, ≈20 px of the other, never both) and the state correlates
exactly with the geometry and with the robot's motion in the RGB panel:

* 15.0 s — operator at working distance, blue set well clear of the capsules →
  **green**;
* 25.0 / 40.0 / 51.5 / 58.3 s — blue set overlapping the capsules, robot
  stationary in the RGB panel → **red**;
* the flip at 34.17 s happens on the frame the two sets touch.

So **green = trajectory verified (robot runs), red = verification failed, the
shield is executing the fail-safe braking trajectory**. 40.9 % of the capture is
red.

## Caveats

* Two moments of the capture are shown, joined by a cross-dissolve (above). The
  shot never reorders events *within* a moment and never alters the shield state.
* The verification loop runs much faster than the 30 fps screen grab, so on the
  boundary the pixel read-out toggles on single frames. The badge therefore
  latches *braking* for 7 frames (0.23 s) and fills verified gaps shorter than
  9 frames — both only ever turn a displayed "verified" into "braking", never
  the other way round, so the badge can never claim a verified trajectory while
  the capsules on screen are red. The 0.3 s of genuine green/red strobe around
  34.2 s is played at 1.0× so it passes quickly; ~7 of the shot's 441 frames
  therefore show a green capsule under a red badge — never the reverse.
* The badge reads `FAILSAFE STOP`, not `FAIL-SAFE STOP` as in the 9.7 s cut, so
  that it matches the author's spelling in the schematic (`failsafe stop`). The
  narration still says "fail-safe stop"; the manuscript writes "failsafe
  trajectory".
* The schematic is a *schematic*: two abstract discs, not a projection of the
  real capsules. It is labelled `Verification` and kept small for that reason.
* The long yellow lines and the small RGB axis triads in the 3-D view are RViz's
  own TF/marker geometry from the recording; they are left untouched.

## Motion discipline

Per the new SPEC §"Motion discipline", every piece of type in this shot enters
on **opacity only, at a fixed position, over ≤ 0.25 s**, and **nothing animates
out**:

| element | before | now |
|---|---|---|
| kicker `Safety verification` | opacity 0.05 → 0.50 s | opacity 0.05 → 0.30 s |
| title `Verify the monitored trajectory` | opacity 0.20 → 0.75 s **and a fade-out over 2.05 → 2.45 s** | opacity 0.20 → 0.45 s, **no exit** — it stays on screen for all 14.7 s |
| `Human reachable occupancy` + caption (and its scrim / leader / dot) | opacity 3.10 → 3.85 s | opacity 3.10 → 3.35 s |
| `Robot reachable occupancy` + caption (and its scrim / leader / dot) | opacity 4.35 → 5.10 s | opacity 4.35 → 4.60 s |
| state badge | opacity 3.00 → 3.55 s | opacity 3.00 → 3.25 s |
| `Verification` schematic panel + labels | opacity 5.70 → 6.35 s | opacity 5.70 → 5.95 s |

`T_TITLE_OUT` is gone from the module. No text in this shot ever moved
positionally, so there was no slide-in to remove.

Motion that **stays**, because it is graphics: the overview figure fading up,
the focus dim, the lit block growing and travelling into the footage, the
cross-dissolve, the live 3-D view, and `mix` — the two schematic discs closing
from disjoint to intersecting as the real sets meet. The badge label and the
schematic's verdict line swap by hard cut when the measured shield state flips,
which is exactly what the rule asks for.

Keeping the title up for the whole shot was checked against the frame: at 96 px
from the left and y 104–160 it sits above and left of the coloured geometry
(union x 418…1287, y 45…752) and clear of the badge, the human callout at
x 1306 and the robot callout at y 782.

---
`_realworld_common.py` was **not** modified by this revision. `_overview.py` was
not modified by this shot either — the widened `REGION_SHIELD` was already in
place when this revision started.

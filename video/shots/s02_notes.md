# s02 — the film's opening title card

> "Our goal: from camera input alone, guarantee the robot is always at a
> complete stop before it could touch the human."

**s02 now plays first**, before s01 (the title/requirement slide) — see the
order in `video/script.py`, which the integrator already changed. It is the
film's title card: the deployment footage fullscreen with the title on top.

Renderer: `video/shots/s02_problem.py` (interp `video`, conda env `chmp-video`).
Shared plumbing: `video/shots/_realworld_common.py` (**unmodified** by this
revision — s08 and s16 were re-rendered anyway, and are unchanged by it).

```bash
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s02_problem.py \
    --out video/build/shots/s02.mp4 --duration 8.5
```

No `--prepare` step and no new packages (cv2 + numpy + Pillow are already in the
env). Render time ~40 s.

## What the shot does

The deployment recording plays **fullscreen** (it fills the whole 1920 × 1080
frame, no card, no border) and the film's title sits over the top of it:

```
      Vision-Based Safe Human–Robot Collaboration     58 px Inter SemiBold, FG
     guaranteeing human safety from camera input alone  34 px Inter Regular, FG_MUTED

  Intel RealSense D435i                               26 px, FG_MUTED, bottom-left
```

The **second title line replaces the standalone goal block** of the earlier cut
("From camera input alone: guarantee the robot is at a complete stop before
contact with the human is possible."). The claim is now made once, in the
title, and the rest of the frame belongs to the footage.

Underneath runs the one passage of the recording in which the operator
**actually puts his hand on the robot**. No pose overlay, no skeleton — the
plain camera image. A ring marks the contact point. The arm is standing still
while he touches it (evidence below), so the footage *is* the claim.

**The kicker is dropped.** SPEC asks every shot for a top-left kicker, but this
is the film's title card and the title has to dominate; a `The goal` kicker
above a 58 px title read as noise. (s01, the other title-ish shot, likewise
carries no chapter kicker.)

Type is Inter throughout (`_realworld_common.font`, `/usr/share/fonts/opentype/inter`).

## Motion discipline

Per the new SPEC §"Motion discipline", **every** piece of type in this shot now
enters on opacity alone, at a fixed position, and never leaves:

| element | before | now |
|---|---|---|
| lead line `From camera input alone:` | faded in over 0.60 s **and slid up 12 px** | gone (folded into the title) |
| statement lines 1 + 2 | faded in over 0.70 s each **and slid up 14 px** | gone (folded into the title) |
| title line 1 | — | opacity 0.10 → 0.35 s, fixed at y = 58 |
| title line 2 | — | opacity 0.32 → 0.57 s, fixed at y = 138 |
| kicker `The goal` | opacity 0.05 → 0.55 s | removed |
| caption `Intel RealSense D435i` | opacity 0.55 → 1.15 s | opacity 0.90 → 1.15 s |
| `contact` pill | faded in **while travelling 22 px down** from the ring | opacity only, fixed at `cy + 62` |

Nothing animates out anywhere in the shot. The contact ring, its halo and the
single expanding pulse keep their motion — they are graphics, and they are the
one thing on screen that is allowed to move besides the footage.

## Source timecodes

`video/media/realworld_source.mp4` — the RViz screen capture of the deployment
(2560×1440, 30 fps, 60.067 s, 1802 frames).

| out | source | speed | what |
|---|---|---|---|
| 0.00 – 5.80 s | **25.80 – 31.60 s** (f 774 – 948) | 1.00× | the operator walks up and reaches for the arm three times (contacts ≈ 27.3, 30.0, 31.5 s) |
| 5.80 – 6.80 s | **31.60 – 32.05 s** (f 948 – 961) | 0.45× | the final reach, slowed |
| 6.80 – 7.30 s | **32.05 – 32.20 s** (f 961 – 966) | 0.30× | the hand settling onto the link |
| 7.30 – 8.50 s | hold on **32.20 s** (f 966) | — | 1.20 s hold on the contact (SPEC wants ≥ 0.6 s) |

The in-point moved 26.35 → **25.80 s** when the duration went 7.8 → 8.5 s, so
the trailing hold stays ~1.2 s instead of growing to 1.75 s. Segments are
`_realworld_common.Timeline` entries and the trailing hold absorbs any further
change of `--duration`; nothing is hardcoded to 8.5 s. Frames are fetched with
`Source.at_blend` so the ramped segments do not judder.

Rejected alternatives: the hand also rests on the arm at 27.25–27.60, 29.9–30.5
and 30.3–30.6 s, but those contacts are shorter and partly motion-blurred; after
**32.3 s** the hand slides back down and leaves the link, which is why the shot
ends at 32.20 s.

## The fullscreen crop (measured on the 2560×1440 capture)

| region | rect `x0,y0,x1,y1` | size |
|---|---|---|
| RViz "RGB Image" panel content (`rw.CROP_RGB`) | `0, 553, 668, 927` | 668 × 374 |
| **what s02 uses** (`CROP`) | `99, 553, 668, 873` | **569 × 320 = 16:9** |

`CROP` is the largest 16:9 sub-rectangle of the RGB panel that still drops the
two dead areas: the black curtain at the far left (panel x < 99) and the empty
floor at the bottom (panel y > 320). It is strictly inside the panel, so no
RViz chrome, options tree, title bar or cursor can ever enter the frame. The
**pose-overlay panel `0, 999, 668, 1373` is deliberately not used** — the author
asked for the skeleton to be left out of this shot.

569 × 320 → **1920 × 1080 with Lanczos (×3.375)**. That is a 3.4× upscale of a
668-px-wide source region and it *is* soft; the instruction was "fullscreen", so
the softness is accepted and managed:

* grade pulled down — `rw.grade(sat=0.82, gamma=1.20, gain=0.90)`;
* a gentle unsharp mask (`sharpen`, amount 0.42, σ 1.8) puts the edges back
  without ringing (checked at 1:1 on the contact region);
* `rw.vignette(0.38)`.

Alternatives considered: the full panel `2,553 → 667,927` (665 × 374, ×2.87 —
less soft but a third of the frame is dead curtain and the operator's head runs
straight into the title), and a tighter `150,573 → 668,864` (518 × 291, ×3.71 —
visibly softer, and the robot's second arm is clipped). `99,553 → 668,873` is
the compromise that was rendered and inspected.

## Legibility of the title over the footage

`rw.top_scrim` ramps from y = 0, which leaves the second line at only ~45 %
darkening — illegible over the lab's whiteboard. The shot therefore has its own
`title_scrim`: **full strength (0.86 toward `style.BG`) down to y = 186**, then
a smoothstep ramp out to y = 330. Title line 1 sits at y = 58 (clear of the
40 px top margin), line 2 at y = 138 with its glyph bottom ≈ 176; the operator's
head enters at y ≈ 224, so the two never touch. `rw.bottom_scrim(806, 0.66)`
keeps the burned-in subtitle band readable and holds the camera caption
(y 866 – 894) well above y = 928.

## The contact point

`CONTACT_SRC = (448, 130)` in **RGB-panel pixels**, unchanged — the *measurement*
is a property of the source, only the mapping to canvas changed:

* old 890×500 card at (515, 362) → canvas **(987, 571)**;
* new fullscreen framing → canvas **(1178, 439)**
  (`(448 − 99)·1920/569 = 1177.6`, `130·1080/320 = 438.8`).

* Measured on **source frame 963 (t = 32.10 s)** by pixel inspection at 10×;
  cross-checked on frames 957, 960 and 966. The operator's fingers lie on the
  white robot link there and the position is stable to ±4 px over
  **31.87 – 32.23 s (frames 956 – 967)**.
* The ring eases in over **source 31.92 → 32.06 s** (output 6.11 → 6.42 s), i.e.
  only while the hand is genuinely at that pixel; it is absent for the first
  three contacts. Nothing is animated that is not in the footage.
* Mark: red (`style.C_DANGER` = contact/danger in the film's semantics) — a soft
  halo ring, a 3 px ring (r ≈ 36 px, scaled up from 30 px for the larger frame),
  a centre dot, one expanding pulse that dissolves, a 2 px leader and a
  `style.BG` pill reading **contact** at a fixed offset of 62 px below the mark.

Supporting read-outs (not shown on screen, but they are why the shot is honest):

* the **arm is motionless** through the contact — mean absolute pixel difference
  of the robot-only region `480,80 → 640,200` (panel coords) against frame 960
  is ≤ 1.2/255 for frames 960 – 980, versus 20–32/255 before frame 945;
* `_realworld_common.display_state()` reads **braking** for every frame in
  940 – 975, and the raw pixel read-out `shield_state()` is an unbroken 1 from
  frame 945 (31.50 s) through 966.
  The robot is already stopped when he touches it — which is exactly the claim
  of the manuscript's §Real-World Deployment ("In all tested instances, the
  robot came to a complete stop before the human operator could reach the
  robot.").

## Verification

`ffprobe`: 1920 × 1080, 30/1, yuv420p, progressive, 255 frames,
**8.500000 s**, no audio stream. Frames extracted at 0.2 / 0.6 / 1.4 / 3.2 /
5.6 / 7.0 / 7.6 / 8.3 s and inspected at 50 % scale plus a 1:1 crop of the
contact region: the title reads, the second line reads over the whiteboard, the
ring lands on the fingers, nothing essential sits below y = 928 or above y = 40.

## Changes from the previous cut (7.8 s "The goal")

* the shot is now the **opening shot of the film** and its **title card**;
* the footage is **fullscreen** (`CROP` 569 × 320 → 1920 × 1080) instead of an
  890 × 500 card;
* the three-line goal block and the `The goal` kicker are gone; the title
  (two lines) replaces them;
* every remaining text entry is **opacity-only ≤ 0.25 s**, no slides, no exits
  (table above);
* in-point 26.35 → 25.80 s, duration 7.8 → **8.5 s**;
* contact marker re-derived for the new framing: canvas (1178, 439), ring
  r 30 → 36 px.

## Caveats

* The upscale is ×3.375 and the image is soft. That is inherent to a 668 × 374
  source region and was accepted per the brief ("it will be soft: grade it down
  and accept the softness … but 'fullscreen' is the instruction").
* The camera is named "Intel RealSense D435i". The manuscript's (commented-out)
  sentence in `content/S4_experiments.tex` writes "Intel RealSense 435i"; D435i
  is the product name of the same device.
* The en dash in `Human–Robot` is U+2013, matching the rest of the film.

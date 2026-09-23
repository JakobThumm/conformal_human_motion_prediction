# s16 — the real-world deployment

Renderer: `video/shots/s16_realworld.py` (interp `video`).
Shared plumbing: `video/shots/_realworld_common.py` (also used by s02 and s08).

```bash
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s16_realworld.py \
    --out video/build/shots/s16.mp4 --duration 21
```

No new packages (cv2 + numpy + Pillow are already in `chmp-video`). Render time
~3.5 min for 720 frames — it decodes and recomposes every frame of the capture
it uses. The one cached artefact is the measured shield state,

```bash
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/_realworld_common.py
# -> video/media/realworld_shield_state.npz   (1802 uint8, ~0.5 kB)
```

which the render loads; it is recomputed automatically if the file is missing.

## Source: `video/media/realworld_source.mp4`

RViz screen capture of the live deployment, 2560 × 1440, 30 fps, 60.067 s,
1802 frames. Layout: options tree + "RGB Image" + "Pose 2D Overlay" panels in a
left column, 3-D view on the right. **None of the RViz chrome is on screen in
the finished shot** — every pixel comes from one of the three crops below.

## Segments chosen (capture timecodes)

| out time | source | speed | content |
|---|---|---|---|
| 0.00 – 4.47 s | **14.60 – 17.95 s** | 0.75× | operator at working distance, trajectory verified |
| 4.47 – 12.47 s | **43.00 – 51.00 s** | 1.00× | braked with the operator close → he clears → the shield re-verifies at 45.77 s → he comes back → it brakes again at **50.67 s** |
| 12.47 – 17.29 s | **53.50 – 57.60 s** | 0.85× | the decisive approach, still verified |
| 17.29 – 18.66 s | **57.60 – 58.97 s** | **1.00×** | the brake onset at 57.83 s — untouched, real time |
| 18.66 – 21.00 s | **58.97 – 60.05 s** | **0.42×** | the operator's hand at the stopped robot (the last ~0.2 s is the held frame, red dot breathing) |

Two 0.42 s **dip-to-canvas** transitions at the cuts (into 43.00 and into
53.50). A cross-dissolve was tried first and rejected: superimposing two cuts
puts red and green capsules — contradictory shield states — in the same frame.
The inset card and the state badge ride the same dip (they are multiplied by
the dip factor `k`), so the badge is fully dark exactly at the midpoint of the
transition, which is where the shield state jumps. The three cuts are the
three clearest approach-and-stop events in the 60 s; 11.5 – 14.5 s is unusable
(the operator dollies the RViz camera inside the occupancy) and is avoided.
In-points were also chosen so the shield state is unambiguous at every cut.

**Playback speed is changed.** Two approach clips are eased (0.75× and 0.85×,
imperceptible on a walking figure) and the last 1.08 s of capture time — the
operator's hand at the already-stopped robot — runs at 0.42×. The brake onset
itself is *not* slowed. No event is re-ordered, cut short or repeated.
Speed-ramped frames are linearly blended (`Source.at_blend`) so the slow motion
is judder-free; at 1.0× from a frame-aligned in-point the blend weight is 0 and
the source frame is untouched. Blending is switched off across a shield state
change, so a red and a green frame are never mixed into a meaningless olive —
the nearest frame is used.

## Crops (measured on the 2560×1440 capture)

| region | rect `x0,y0,x1,y1` | size | used as |
|---|---|---|---|
| 3-D render area (full) | `692, 98, 2542, 1414` | 1850 × 1316 | — |
| **this shot's 3-D crop** | `900, 400, 2500, 1300` | 1600 × 900 | main image, Lanczos **×1.20** → 1920 × 1080 |
| "Pose 2D Overlay" content | `0, 999, 668, 1373` | 668 × 374 | inset card, Lanczos ×0.78 → 520 × 292 |
| "RGB Image" content | `0, 553, 668, 927` | 668 × 374 | (used by s02) |

Composition: the 3-D crop is placed so the robot base and the operator's
occupancy sit in the left ~60 % of the frame; the inset card lives in the empty
upper right at `x 1304…1824, y 140…432`. Everything typographic is above
`y = 928` (subtitle safe area): kicker at y 58, badge at y 46…104,
equipment strip at y 846…902.  The card itself ends at y = 432 and now carries
no caption.

## Motion discipline

Per the new SPEC §"Motion discipline", every overlay in this shot enters on
**opacity only, at a fixed position, over ≤ 0.25 s**, and **nothing animates
out**:

| element | before | now |
|---|---|---|
| kicker `Results · real-world deployment` | opacity 0.00 → 0.45 s | opacity 0.00 → 0.25 s |
| equipment strip (pill) | opacity 0.30 → 1.00 s | opacity 0.30 → 0.55 s |
| 2-D pose inset card | opacity 0.55 → 1.25 s | opacity 0.55 → 0.80 s |
| state badge | opacity 0.80 → 1.40 s, **plus its own `badge_gate()` fade-out / fade-in (0.18 s ramps) around every cut** | opacity 0.80 → 1.05 s; `badge_gate` **deleted** — the badge is multiplied by the dip factor `k`, i.e. it rides the picture's own dip-to-canvas |

`badge_gate()` was the only exit animation in the shot and is gone from the
module. Nothing in s16 ever moved positionally, so there was no slide-in to
remove.

Motion that **stays**, because it is graphics: the footage itself, the two
dip-to-canvas transitions between cuts (explicitly allowed by the author), and
the slow breath on the badge's status dot over the held final frame.

Removing `badge_gate` is also *safer*, not just tidier: the gate blanked the
badge over a hand-tuned window around each cut, whereas `k` is zero exactly at
the frame where `ts_now` jumps from the outgoing clip to the incoming one, so
the badge can never carry the previous clip's shield state into the next one.

## Treatment

`_realworld_common.grade()`: mild desaturation (×0.86), gamma 1.24, gain 0.93 and
the black point lifted to `style.BG` (#0E1117), so the RViz grey background melts
into the film canvas while the blue occupancy and the red/green capsules keep
their punch. Plus a vignette and top/bottom scrims (the bottom scrim also
guarantees the burned-in subtitles read).

**The shot is silent.** `common.FrameWriter` encodes with `-an`, so `s16.mp4`
carries a video stream only — there is no audio anywhere in this renderer, and
none of the capture's own audio is ever read (`_realworld_common.Source`
decodes video frames with cv2). Any lab ambience under this shot comes from the
film's audio mix in `video/build.py::build_audio`, which ducks
`video/media/realworld_source.mp4`'s own track in at volume 0.12; that file is
the integrator's and is not touched here.

Overlays, all in `video/style.py` colours:

* kicker `6 · Real-world deployment` (top-left);
* persistent equipment strip (bottom-left pill):
  **Franka Emika · Intel RealSense D435i · 25 Hz · SARA shield**;
* inset card of the live 2-D pose — **uncaptioned** (the equipment strip
  already names the RealSense D435i, and a second label under the card only
  crowded the upper right);
* a state badge, top-right: `VERIFIED` (green) / `FAIL-SAFE STOP` (red).

## The state badge is measured, not scripted

`_realworld_common.compute_shield_state()` counts strongly-red vs strongly-green
pixels inside the 3-D view for all 1802 frames and writes the result to the npz.
Red and green never coexist (≈600 px of one against ≈20 px of the other), so the
read-out is exact. 40.9 % of the capture is red.

**What the colours mean:** the robot's reachable-occupancy capsules are
**green while SARA shield has verified the current trajectory** and **red while
verification fails and the robot is executing its fail-safe braking
trajectory**. Evidence: at 15.0 s the operator is at working distance, the blue
set is well clear of the capsules and they are green; at 25.0 / 40.0 / 51.5 /
58.3 s the blue set overlaps the capsules, the capsules are red and the arm is
stationary in the RGB panel; the flips coincide frame-for-frame with the two
sets touching. The human's conformal occupancy is blue throughout and never
changes colour.

## Caveats

* **Display debounce.** The verification loop runs far faster than the 30 fps
  screen grab, so at the boundary the pixel read-out toggles on single frames.
  The badge latches *braking* for 7 frames (0.23 s) and fills verified gaps
  shorter than 9 frames (0.3 s). Both operations can only turn a displayed
  "verified" into "braking", never the reverse, so the badge never claims a
  verified trajectory while the capsules on screen are red. Raw, unfiltered
  state is what is stored in the npz. In practice the shield chatters on the
  containment boundary for ~0.3 s around each onset and the capsules genuinely
  strobe green/red in the recording; **36 of the shot's 720 frames** therefore
  show a green capsule under a red badge (never the reverse). Those stretches
  are deliberately played at 1.0× so they pass quickly.
* `25 Hz` is the pipeline rate implied by the paper's `K_I = 50` input frames =
  2 s of history; the screen capture itself is 30 fps, and RViz's own fps
  read-out (which is cropped away) fluctuates between 18 and 31.
* The camera is written "Intel RealSense D435i"; the manuscript's commented-out
  sentence in `content/S4_experiments.tex` writes "Intel RealSense 435i" — same
  device, D435i is the product name.
* The claim in the narration ("in every tested instance the robot stood still
  long before the operator could reach it") is the manuscript's sentence at the
  end of `content/S4_experiments.tex` §"Real-World Deployment"; this shot shows
  three of those instances and does not put a number on screen.
* The long yellow lines and the small RGB axis triads in the 3-D view are RViz's
  own TF/marker geometry from the recording, left untouched.

## Verification (after the motion-discipline revision)

`ffprobe`: 1920×1080, 30/1, yuv420p, progressive, 630 frames, 21.000000 s and
**no audio stream**.  Frames inspected across all three cuts, both shield
states and both dip transitions; nothing essential below y = 928 px or above
y = 40 px, and no overlay is ever left floating on the canvas mid-dip.

---
New files added by these three shots: `video/shots/_realworld_common.py` (shared
plumbing for s02 / s08 / s16) and the cached read-out
`video/media/realworld_shield_state.npz` (429 bytes). Nothing else under `video/`
was touched.

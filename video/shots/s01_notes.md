# s01 — failure vs dangerous failure · the 10⁻⁶ / h target

**Duration** 14.5 s · **interp** `video` (manim 0.20) · **file** `video/shots/s01_title.py`

```bash
# optional, only if video/media/s01_frame.png is missing or the source recording changes
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s01_title.py --prepare
# render (main() also runs prepare automatically when the still is missing)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s01_title.py \
    --out video/build/shots/s01.mp4 --duration 14.5
```

Narration (13.32 s of audio, `video/build/audio/s01.wav`, starts at 0.15 s):
*"Cameras drop frames. Predictions go wrong. That is a failure. It turns dangerous only if the
robot then touches the person. Standards cap dangerous failures at one per million hours: one in
a hundred and fourteen years of round-the-clock work."*

## 2026-09-22 revision: the title card is gone, and words no longer move

Two changes, both from the author.

1. **The film's title moved to the new opening shot (s02, which now plays before s01).** s01
   therefore **drops its title card entirely** — no "Vision-Based Safe Human–Robot Collaboration",
   no subtitle, no title-becomes-a-header transform. The shot opens directly on the *Failure*
   pictogram. The graded deployment still stays, but it is now **static at 8.5 %** from frame 0
   (constant `BACKDROP_OPACITY`) instead of crossfading 0 → 40 % → 8.5 % to carry the title; set
   it to `0.0` for a pure-black card.
2. **`SPEC.md` "Motion discipline".** Every word, number, caption and legend enters on **opacity
   only**, in ≤ 0.25 s, and **nothing ever exits**. Converted or deleted here:

   | was | now |
   |---|---|
   | `FadeIn(title, shift=DOWN*0.22)` + `img.animate.set_opacity(0.40)` | deleted (no title) |
   | `FadeIn(sub, shift=DOWN*0.15)` | deleted (no subtitle) |
   | `title.animate.scale(0.78).move_to(...)`, `FadeOut(sub)`, `img.animate.set_opacity(0.085)` | deleted; backdrop is static |
   | `FadeIn(left["pic"], shift=UP*0.10)` | `FadeIn(...)`, no shift |
   | `FadeIn(chip, shift=UP*0.10)` (both cards) | `FadeIn(chip)`, 0.22 s, split out of the `GrowFromCenter` play so the word does not fade over 0.5 s |
   | `FadeIn(VGroup(pic, esc), shift=UP*0.10)` | two plain `FadeIn`s |
   | `FadeIn(implies, shift=UP*0.08)`, 0.50 s | `FadeIn(implies)`, 0.25 s |
   | `FadeOut(beat2)` (the whole F/D block) | `self.remove(*beat1)` — **hard cut** |
   | `FadeIn(band, shift=UP*0.12)` ×5 in the `LaggedStart` | same lagged fade, no shift |
   | `FadeIn(ax.ticks, shift=UP*0.06)`, `FadeIn(req, shift=UP*0.08)`, `FadeIn(years, shift=UP*0.10)` | plain `FadeIn`, 0.22–0.25 s |

   Kept, because they are **graphics**: the escape ring and the impact burst growing
   (`GrowFromCenter`), the robot arm being drawn (`Create`), the dashed 10⁻⁶ rule being drawn,
   the green wash, and PL d's fill lighting up.

   ⚠ manim's Cairo `Scene.remove` calls `restructure_mobjects(..., extract_families=False)`, so
   it only drops the **exact** objects that were added — handing it a parent `VGroup` whose
   children were added individually is a silent no-op. `beat1` is therefore a flat list.

3. **Layout.** With no header at the top of the frame the PL-axis beat was bottom-heavy, so the
   whole beat is lifted by `AX_DY = 0.35` manim units (≈ 47 px). The F/D cards were already
   optically centred and are unchanged.

## Beats (14.5 s)

| t (s) | what |
|---|---|
| 0.00 | kicker + the static 8.5 % backdrop are simply *there* on frame 0 |
| 0.20–0.75 | left card: green stick human + blue dashed conformal sets |
| 1.05–1.72 | the reaching hand is marked red (outside its set), then the chip **Failure** |
| 1.85–2.07 | its caption — and then **1.4 s of nothing but the Failure panel** |
| 3.45–4.80 | “+”, right card = the same pictogram, then the robot arm is *drawn* reaching in |
| 5.05–5.72 | red impact burst on that same hand, then the chip **Dangerous failure** |
| 5.85–6.07 | its caption |
| 6.40–6.65 | `dangerous failure = failure + robot contact` |
| 7.20 | **hard cut**: the F/D block is removed; axis title + ISO caption fade up |
| 7.65–8.60 | the five PL bands stagger in **a → e, left → right** |
| 8.85–9.10 | decade ticks + the “← more dangerous failures / fewer →” direction captions |
| 9.45–10.37 | dashed 10⁻⁶ rule, green wash over the compliant half, PL d lights up, “required: fewer than 10⁻⁶ per hour” |
| 10.45–10.70 | **“that is one dangerous failure every 114 years”** |
| 11.70–11.95 | the arithmetic line, then a **2.55 s hold** |

### Where the reclaimed ~2.7 s went

The title card used to own 0.00–2.70 s, i.e. the whole first narration sentence. Now:

* the **Failure** panel is complete at **2.07 s** instead of 4.10 s, so it reads on its own for
  **1.38 s** before the robot arrives (it used to get 0.15 s);
* the **Dangerous failure** panel is complete at **6.07 s** instead of 7.00 s, and the
  `D = F + contact` line lands 0.2 s earlier;
* the PL-axis beat keeps its narration-locked cues (7.20 / 8.85 / 9.45 / 10.45) but every fade is
  shorter, so each state is static for longer, and the arithmetic line lands 0.2 s earlier —
  the closing hold on the finished frame grows from 2.05 s to **2.55 s**.

Cue times are matched against the synthesised narration (sentence ends measured at ≈ 3.4 / 6.9 /
10.1 / 13.5 s; the only detectable pause in the wav is the colon at 9.91 s + 0.15 s offset).

## Content provenance

* **Title / authors / venue** — **not in this shot any more.** The title card is s02's job; the
  submission is anonymous (`main.tex` sets `\setboolean{anon}{true}`) so no authors or venue
  appear anywhere in the film.
* **F vs D** — `content/S3_methodology.tex` §"Probability of Dangerous Failures Per Hour":
  *F prediction failure*: `∃ j,k : p^j_k ∉ S^j_k`; *D dangerous failure*: "a contact occurs
  although c_safe = 1"; and "D ⇒ F". The two panels encode that implication literally — the right
  panel **is** the left panel plus the robot, and the joint the robot touches is the same joint
  that escaped its set. The bottom line spells it out as `D = F + contact`.
* **PL bands** — `video/shots/_pl_axis.py::PL_BANDS`, a copy of the repo's own classifier
  `src/.../generate_plots/conformal_results_common.py::PL_BANDS`
  (a 1e-5–1e-4, b 3e-6–1e-5, c 1e-6–3e-6, d 1e-7–1e-6, e 1e-8–1e-7).
  S1 states PL d = `1e-7 ≤ PFH_D < 1e-6` and "collaborative robot applications commonly target
  PL d as specified in ISO 10218-1:2021" — the caption under the axis title quotes that.
* **114 years** — computed in the module, never hardcoded:
  `HOURS_PER_YEAR = 24 × 365.25 = 8766`, `CAP_HOURS = 1 / 1e-6 = 1e6`,
  `CAP_YEARS = 1e6 / 8766 = 114.077…` → headline "114 years", arithmetic line "114.1 years".
  The on-screen line prints the whole chain so the number is checkable:
  `one in 1 000 000 hours · one year of round-the-clock work = 8 766 hours (24 h × 365.25 d) ·
  1 000 000 / 8 766 = 114.1 years`.
* **Backdrop** — frame 1300 of `video/media/realworld_source.mp4` (the real deployment capture),
  cropped to the RViz "Pose 2D Overlay" panel (`crop=648:364:13:988`), blurred, graded down and
  vignetted into `video/media/s01_frame.png` (0.93 MB).

## Colour use in this shot

Per `SPEC.md`: green = the true human, blue = our conformal prediction sets, light grey = robot,
red = contact / dangerous event. The *plain* "Failure" chip uses **YELLOW**, i.e. a warning rather
than a danger — red is reserved for the dangerous failure so that the two chips are instantly
distinguishable. Orange is not used here (it belongs to the ISO 13855 baseline).

## Shared helper `video/shots/_pl_axis.py` (owned by the s01/s06/s14/s17 group)

* `txt/mtxt/kicker/fs` — style.py **pixel** sizes → manim `font_size`. `txt`/`mtxt` render at
  `RENDER_FS = 48` and scale down, because below font_size ≈ 40 manim collapses word spaces.
  **Never construct `Text`/`MarkupText` directly in a shot.** `MathTex` is the only allowed
  second typeface (math notation only).
* `PLAxis(center, width, bar_h, target)` → `.group`, `.bands[name]`, `.labels[name]`, `.bar`,
  `.ticks`, `.x_of(pfh)`, `.top_of()`, `.bottom_of()`, `.threshold(pfh)`, `.marker(pfh, color)`,
  and (new) `.region(lo, hi)` + `.direction_labels()`.
* `TimedScene` — `pl()/hold()/at()/finish()` keep the scene length equal to `--duration`
  (`finish()` over-runs by 0.25 s so `common.conform`'s `-t` trim lands on the exact frame count).
  Note that `pl()` takes its bookkeeping from the `run_time=` **keyword**, which manim also
  applies to every animation in the call — so a graphic and a word cannot share one `pl()` if
  they need different durations. Under the motion rule that is the normal case: issue the
  graphic's `pl(..., run_time=0.7)` and the label's `pl(FadeIn(lab), run_time=0.22)` separately.

**`_pl_axis.py` was not modified by the 2026-09-22 motion pass** — the rule is about how the
shots *use* it, not about the helper.

### 2026-09 change: the axis now runs PL a → PL e (author request)

* `PL_BANDS` is listed **a first**, and `x_of()` maps high PFH_D to the **left**
  (`x = center + (0.5 − f) · width`), so the decades read 10⁻⁴ … 10⁻⁸ left → right.
* Because that direction is counter-intuitive, `direction_labels()` returns the pair
  "← more dangerous failures" / "fewer dangerous failures →". It is deliberately **not** part of
  `.group`: s14 has no vertical room under its ticks.
* Colour: instead of colouring only the target band, every band's fill is blended towards
  `style.GREEN` along `BAND_RAMP` (a 0.00 → e 0.30) with the label colour ramping
  `FG_MUTED → GREEN` (`LABEL_RAMP`), so **PL e is the green end**. The `target` band (default
  "d") is kept identifiable by a **green 2.4 px outline** plus the brighter label, not by more
  fill — s01 additionally raises its fill to green @ 0.44 on the reveal, s14 to @ 0.34 and drops
  its result marker on it.
* `region(lo, hi)` is used by s01 for the translucent green wash over the compliant half
  (PFH_D < 10⁻⁶ = PL d + PL e).
* **s14 re-checked** after the change:
  `.../chmp-video/bin/python video/shots/s14_pfhd.py --out /tmp/s14.mp4 --duration 11.3` →
  1920×1080, 30 fps, yuv420p, 11.300000 s; the last frame shows the 9.50 × 10⁻⁷ marker landing
  just inside the highlighted PL d band, immediately right of the 10⁻⁶ tick. Nothing else in s14
  touches the axis, so no edit to `s14_pfhd.py` was needed.

## Verification (2026-09-22 re-render)

`ffprobe`: 1920×1080, 30/1 fps, yuv420p, progressive, **435 frames, 14.500000 s**, no audio.
Frames inspected at t ≈ 0.6 / 1.6 / 2.6 / 4.2 / 5.4 / 6.2 / 7.0 / 7.16 / 7.25 / 8.2 / 9.9 /
11.0 / 12.2 / 14.4 s — including the two frames either side of the 7.20 s hard cut, which is
clean (7.16 s = the complete F/D block, 7.25 s = bare backdrop + the axis title fading up).
With entry-only opacity the frames read as a sequence of *stable states* rather than a sequence
of transitions.

Safe area checked over **all 435 decoded frames**: the brightest pixel above y = 40 px is 26/255
and below y = 928 px is 26/255 (both are the 8.5 % backdrop); zero pixels > 90/255 in either
band. Drawn-ink extent (> 90/255) over the whole shot: **rows 57 … 854**.

## Caveats / things the integrator may want to change

* The backdrop stays visible at 8.5 % for the whole shot. Set `BACKDROP_OPACITY = 0.0` (module
  level) for a pure-black card.
* The shot ends on a 2.55 s hold of the full final frame (axis + 114 years). If the cut feels
  slow, move the `self.at(11.70)` arithmetic cue later rather than shortening the hold.
* **The title now has to be carried by s02.** As of this revision `video/shots/s02_problem.py`
  shows the kicker "The goal" and the goal statement, but *not* the paper title — if s02 is not
  also updated, the film has no title card at all.

# s17 — closing card

**Duration** 5.2 s · **interp** `video` (manim 0.20) · **file** `video/shots/s17_outro.py`

```bash
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s17_outro.py \
    --out video/build/shots/s17.mp4 --duration 5.2
```

Static card: paper title and three takeaways with semantic colour keys (blue = our sets,
green = the certified number, purple = OOD). Everything is on screen by t = 1.22 s and holds for
the remaining 3.98 s.

## 2026-09-22 revision: motion discipline (content unchanged)

The card is nothing but words, so under `SPEC.md` "Motion discipline" it is now **pure opacity**:

| was | now |
|---|---|
| `FadeIn(title, shift=DOWN*0.15)`, 0.8 s | `FadeIn(title)`, 0.25 s |
| `FadeIn(row, shift=RIGHT*0.20)` ×3, 0.45 s each | `FadeIn(row)`, 0.22 s each, 0.10 s apart |
| `FadeIn(url)`, 0.6 s (`--show-url` only) | `FadeIn(url)`, 0.22 s |

Nothing exits; there was never an exit here. The 0.93 s of animation time that the slide-ins
cost is now hold: the finished card is static from 1.22 s instead of 2.15 s.

## The closing line is gone

The card used to end on a fourth line: the project URL, or — because `main.tex` compiles with
`\setboolean{anon}{true}` — the anonymous stand-in "code, models and data in the supplementary
material". **That line has been removed entirely**: `URL_ANON` is deleted, and the default
(anonymous) cut simply ends on the three takeaways.

`--show-url` still works and is still the camera-ready switch. The layout is picked from the
flag, so both cuts stay balanced:

| | title *y* | first row *y* | row pitch | URL *y* |
|---|---|---|---|---|
| default (no URL) | 2.00 | 0.50 | 0.95 | — |
| `--show-url` | 2.35 | 0.85 | 0.85 | −2.05 |

Without the URL the block is re-centred slightly above optical centre (the burned-in subtitle
owns the bottom band) and the rows breathe wider; the lowest element is the third takeaway at
y ≈ 745 px, well clear of the 928 px subtitle line.

## Numbers and where they come from

| line | source |
|---|---|
| "7.6× smaller conformal prediction sets · 1−ε = 99.99 %" | S4: "a 7.6 times smaller median volume than the sets from the ISO 13855 model"; median 0.090 m³ vs 0.686 m³ (`results/final/conformal_prediction_sets/coverage_stats_*.csv`); 1−ε from S4's protocol paragraph |
| "PFH_D ≤ 9.5 × 10⁻⁷ per hour · 99.999 % confidence" | same CSV row as s14 (`pfh_d_upper` = 9.50193e-7, `epsilon_d` = 1e-5) |
| "24 % fewer interrupted predictions from OOD handling" | S4 / `results/final/full_pipeline/`: invalid pose buffers 16.57 % → 12.63 % |
| URL `jakob-thumm.com/conformal_human_motion_prediction` (only with `--show-url`) | `content/A0_abstract.tex`, non-anonymous branch |

## Two deliberate deviations from the brief

1. **7.6×, not 8×.** The abstract and S1 round the volume ratio to "8 times"; S4, the CSVs and the
   film's own s10 narration ("seven point six times smaller") say 7.6. Using 8× here would
   contradict a line the audience heard 40 s earlier, so the card says 7.6×. Change the first
   entry of `TAKEAWAYS` if the abstract's rounding should win.
2. **The URL de-anonymises the submission**, so it is **off** by default: `main.tex` compiles
   with `\setboolean{anon}{true}` and the abstract's anon branch says the code link is "omitted
   here for anonymity". Pass `--show-url` (or set `args=["--show-url"]` on shot s17 in
   `video/script.py`) for the camera-ready cut.

## Verification (2026-09-22 re-render)

`ffprobe`: 1920×1080, 30/1, yuv420p, progressive, **156 frames, 5.200000 s**, no audio stream —
for both the default cut and `--show-url`. Frames inspected at t ≈ 0.1 / 0.3 / 0.55 / 0.8 / 1.0 /
1.3 / 2.5 / 5.1 s, at 50 % scale; they read as four stable states (empty → title → +row → …).
Safe area measured over **all 156 decoded frames**: max luminance above y = 40 px and below
y = 928 px is 16/255, zero pixels > 90/255 in either band. Drawn-ink extent **rows 57 … 745**
(default) / **57 … 831** (`--show-url`).

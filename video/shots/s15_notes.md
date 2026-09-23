# s15 — Pipeline results (OOD-handling ablation + runtime)

**Module** `video/shots/s15_ood_results.py` · interp `video` · planned 7.0 s (duration-agnostic).
Kicker: **`Pipeline results`**.

## Build

```bash
cd /home/thumm/code/conformal_human_motion_prediction
# 1) cache — parses the two committed result files; no GPU, either interpreter works
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s15_ood_results.py --prepare
#    -> video/media/s15_results.npz (1.6 kB)
# 2) render (~10 s)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s15_ood_results.py \
    --out video/build/shots/s15.mp4 --duration 7.0
```

No new packages; PIL at 2× supersampling (matplotlib is already in `chmp-video` and is only
used head-less, for mathtext — see "Typography" below).

## Typography — real maths, no LaTeX subprocess

Every symbol on the slide is typeset, not faked with Unicode subscripts:

| on screen | source |
|---|---|
| *N*<sub>req</sub> (ablation legend, both states) | `$N_{\mathrm{req}}$` |
| *f*<sub>2D</sub>, SLU<sub>2D</sub>, *f*<sub>mot</sub>, SLU<sub>mot</sub> (runtime legend) | `$f_{\mathrm{2D}}$`, `$\mathrm{SLU}_{\mathrm{2D}}$`, `$f_{\mathrm{mot}}$`, `$\mathrm{SLU}_{\mathrm{mot}}$` |

which is exactly how `content/S3_methodology.tex` and `figures/result_tables/runtime.tex` write
them. The renderer is PIL, so the maths comes from matplotlib's
`mathtext.MathTextParser("agg")` — Computer Modern glyph coverage plus the typeset box depth,
no LaTeX subprocess, deterministic. **This is the helper from `s05_motion.py`** (`_math_raster`
/ `math_glyph`), copied in and adapted to this shot's supersampled `Canvas` rather than promoted
to a shared module. It measures the Inter cap height of the line it sits in, picks the dpi that
renders a Computer Modern "K" to the same cap height ~4× oversampled, and pastes the downsampled
coverage onto that line's *measured* Inter baseline.

`Canvas.rich(xy, runs, ...)` draws a mixed line: a run starting with `"$"` is maths, anything
else is Inter, and it returns the advance width so callers can lay out what follows. `STAGES`
therefore stores label **run lists** instead of strings; `plain()` flattens them to ASCII for the
`.npz` cache and the `--prepare` printout only.

Inter remains the only sans on the slide (SPEC "Type": maths is the single permitted exception).

## Panel (a) — the fallback

Three dumbbells (before → after per item, the canonical form for this comparison), each on its
own labelled mini-scale with both endpoints printed, plus a relative-change chip. Parsed at
prepare time from `results/final/full_pipeline/full_pipeline_results.tex` (byte-identical to
manuscript `figures/result_tables/full_pipeline_results.tex`):

| | N_req = 50 | N_req = 3 (ours) | chip |
|---|---|---|---|
| invalid pose buffers | 16.57 % | 12.63 % | −23.8 % |
| motion prediction valid | 68.70 % | 74.74 % | +8.8 % |
| MPJPE | 54.14 mm | 55.15 mm | +1.9 % |

−23.8 % and +1.9 % are exactly the numbers in
`results/final/full_pipeline/full_pipeline_sentence.tex`; the narration's "twenty-four percent"
is that figure rounded. The cost row (MPJPE) is chipped in `YELLOW`, the two wins in `C_TRUTH`
green, so the trade-off is not hidden.

### ⚠ Caveat — the repo CSVs for N_req = 3 and 5 have moved since the table was generated

`full_pipeline_results.tex` (and the manuscript table) were written on **17 Jul**. The underlying
`results/final/full_pipeline/n_correct_poses_required_{3,5}/motion_validity_stats.csv` and
`.../motion_prediction/mpjpe_results_test.csv` were **re-run on 14 Sep** and now read:

| | current CSV (14 Sep) | table / video (17 Jul) |
|---|---|---|
| N_req = 3, invalid pose buffers | 9.78 % | 12.63 % |
| N_req = 3, motion valid | 82.74 % | 74.74 % |
| N_req = 3, MPJPE | 55.83 mm | 55.15 mm |

(the N_req = 10 and 50 rows are unchanged and still match the table.) Re-running
`python -m conformal_human_motion_prediction.generate_plots.generate_full_pipeline_results`
would therefore change the table to −41 % invalid pose buffers. **The shot deliberately uses the
committed table**, so the film and the submitted manuscript carry the same digits; if the table
is regenerated before submission, re-run `--prepare` and the shot follows automatically (it parses
the `.tex`, it does not hardcode).

## Panel (b) — the clock

Stacked horizontal bar of the per-stage **medians** from
`results/final/runtime/runtime_stages.csv` (= manuscript `figures/result_tables/runtime.tex`),
450 pipeline steps, one NVIDIA RTX 5090:

*f*<sub>2D</sub> 13.20 · SLU<sub>2D</sub> 36.39 · triangulation 1.87 · *f*<sub>mot</sub> 4.69 ·
SLU<sub>mot</sub> 9.08 · sets 0.64 → **65.86 ms** (the six medians sum to the reported total
exactly), 15 Hz.

Colour carries meaning rather than identity: both sketched-Lanczos monitors are `C_OOD` purple,
both network forward passes `C_PRED` teal, triangulation `C_ROBOT` grey, set computation
`C_OURS` blue. The two purple segments are underlined and called out as
**45.47 ms = 69 % of the step**, which is the claim in
`results/final/runtime/runtime_sentence.tex`. Segment order is pipeline order (never reordered to
group the purples), and the six values are also given in a legend table because the smallest
segment (0.64 ms) is only ~8 px wide.

## Verification (after the "Pipeline results" revision)

`ffprobe`: 1920×1080, 30/1, yuv420p, progressive, 210 frames, 7.000000 s, no audio stream.
Frames inspected at t ≈ 0.4 / 1.2 / 2.4 / 3.4 / 4.4 / 5.4 / 6.9 s, plus the finished frame at
50 % scale: the maths sits on the Inter baseline in every line, the ablation legend
(bumped 23 → 25 px so the lighter Computer Modern face still reads at half scale) clears the
first dumbbell, and the lowest element is the `60 mm` scale label at y ≈ 800 px.

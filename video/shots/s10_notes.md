# s10 — Prediction-set volume vs ISO 13855

**Module** `video/shots/s10_volume_chart.py` · interp `video` · planned 9.0 s (duration-agnostic).

## Build

```bash
cd /home/thumm/code/conformal_human_motion_prediction
# 1) cache — reloads the 500 MB test cloudpickle and recomputes both radius fields (~40 s, CPU)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu \
  .venv/bin/python video/shots/s10_volume_chart.py --prepare   # -> video/media/s10_volume.npz (4 kB)
# 2) render (~13 s)
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s10_volume_chart.py \
    --out video/build/shots/s10.mp4 --duration 9.0
```

No new packages; PIL at 2× supersampling.

## Chart form (dataviz skill)

The job is *compare two distributions* + *one headline*. Form: two stacked ridge densities on a
shared log-volume axis (not two bars — two bars would throw away the 7.7 M-sample shape and the
heavy tail), an emphasis pair of categorical hues fixed by the film's semantics
(`C_ISO` orange = baseline, `C_OURS` blue = ours), direct labels on both medians, 5th–95th
whiskers under each ridge, and the ratio as a hero stat tile at the top right so the point lands
in ~1 s. The second panel is a two-dot log dot-plot, because a miss-rate spanning 1.5 decades is
a position-on-log-scale comparison, not a bar. No dual axis anywhere; the two panels are
separated by a rule.

The ISO ridge is visibly multi-modal: that is real structure — its radius is
`input_uncertainty + v_h·t` over the 10 discrete horizon steps, so the distribution is a mixture
of 10 shifted components.

## Provenance — the distributions are recomputed, not read off a table

`--prepare` reloads
`results/final/conformal_prediction_sets/motion_prediction_results_test.cloudpickle`
(59 472 × 10 × 13 predictions / targets / covariances / last-input poses) and rebuilds both sets
exactly the way `examples/motion_prediction.py` does:

* ours: `inference_helper.conformal_set_radius(model_cov, input_cov, calibrator)` with
  `models/motion_prediction/conformal_calibration/conformal_calibrator.npz` (conditional
  conformal, `level = 0.9999`), `V = 4/3 π r³`;
* ISO 13855: `eval_utils.compute_sara_predictions(last_input_poses, t, V_HUMAN_ISO = 2.0 m/s,
  input_uncertainty)`;
* padding masked exactly like `eval_utils.simple_coverage_stats_sara` → 7 731 360 valid spheres.

Reproduced percentiles (identical to 6 dp with the committed CSVs and manuscript Table I):

| | 5 % | 50 % | 95 % | source CSV |
|---|---|---|---|---|
| ours | 0.015309 | 0.090136 | 0.653729 | `coverage_stats_conformal_prediction_sets.csv` |
| ISO 13855 | 0.016880 | 0.685770 | 3.243962 | `coverage_stats_sara.csv` |

Ratio of medians = **7.6×** (computed live in the shot, not hardcoded).

## Miss-rate panel — one deliberate rounding

The live recomputation gives **1.626 × 10⁻⁴** (ours, 1 257 escaped spheres) and
**4.527 × 10⁻⁶** (ISO, 35 escaped spheres). The shot prints the values **as the manuscript prints
them** — `1.6 × 10⁻⁴` and `5.0 × 10⁻⁶` — because the table derives the ISO figure from the
4-decimal coverage in the CSV (99.9995 %). The renderer asserts the live values agree with the
printed ones within rounding, so the two can never silently drift apart. Caveat worth knowing:
the exact ISO miss-rate is 4.5 × 10⁻⁶, not 5.0 × 10⁻⁶.

`PFH_D` footnote values (9.50 × 10⁻⁷ ours, 1.62 × 10⁻⁷ ISO, per hour) are read from manuscript
`figures/result_tables/all_conformal_prediction_results.tex`.

## Honesty

The shot says in as many words: *"Ours misses more often."* and then *"Both clear the safety bar"*
with the two PFH_D numbers, which is the paper's actual claim (H1 + H2). It does not imply the
miss-rate itself is the certified quantity.

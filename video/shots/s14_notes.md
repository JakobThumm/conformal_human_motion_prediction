# s14 — the PFH_D bound, term by term

**Duration** 13.5 s · **interp** `video` (manim 0.20) · **file** `video/shots/s14_pfhd.py`

```bash
/home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s14_pfhd.py \
    --out video/build/shots/s14.mp4 --duration 13.5
```

No prepare step: the shot reads
`results/final/robot_shield_risk_volume/shield_risk_volume_results.csv` (10 rows, ~8 kB) at render
time, **re-derives both Clopper-Pearson factors with `scipy.stats.beta.ppf`**, and **asserts** that
the three factors multiply to the CSV's own `pfh_d_upper` before drawing anything. If someone
re-runs the simulation the shot follows automatically, and if the bound stops reproducing the shot
fails loudly instead of drawing a number it cannot derive.

## Revision (2026-09-22): the paper's simplified, three-factor bound

Earlier cuts assembled **four** factors, splitting `P(D|F)` into the geometric witness-region term
`P(W|F) = V(W)/V(F)` times `P(D|F,W)` over the 4 × 10⁹ trials — the way the simulation internally
computes it. On the author's request the shot now shows eq. `pfh_bound` of
`content/S3_methodology.tex` **verbatim**:

```
PFH_D  ≤  N_h · B⁻¹(1−ε_F;   k_eff+1, N_eff−k_eff)
              · B⁻¹(1−ε_D|F; k_D+1,   N_D−k_D)
```

The `W` term is gone from the screen entirely, and the last factor is `P(D|F)` over `N_D`
**uniform** placements. This matches s12, which no longer uses witness-region language either.
The CSV's `p_d_given_f_upper` is stored as `p_w · p_d_given_f_w_upper`, but that product is
numerically the same number as the paper's plain Clopper-Pearson bound over
`equivalent_uniform_placements` (verified below), so nothing on screen is a re-derivation: it *is*
the CSV column.

**s13 (the dangerous-trial replay) was cut**, so s14 is the only place the viewer meets
`k_D = 4`. The shot therefore opens on a dedicated `4` beat (≈ 5 s, the whole first narration
sentence) and the same red `4` is then the first glyph of the third factor's caption, so the
number the bound rests on is never just asserted.

## 2026-09-22 revision: motion discipline (content unchanged)

Per `SPEC.md` "Motion discipline" — words, numbers and equations enter on **opacity only**
(≤ 0.25 s) and **nothing exits**. Converted or deleted:

| was | now |
|---|---|
| `FadeIn(big4, scale=0.72)`, 0.65 s, `ease_out_cubic` — the `4` scaling in | `FadeIn(big4)`, 0.25 s (a numeral is a word, not a picture) |
| `FadeIn(l1/l2, shift=UP*0.10)`, 0.60–0.65 s | plain `FadeIn`, 0.25 s |
| `FadeIn(l3)`, 0.60 s | 0.25 s |
| **`ReplacementTransform(big4, k4)`** — the `4` flying into factor 3's caption | **deleted**: `self.remove(big4, l1, l2, l3)` (hard cut) and the caption, red `4` and all, is simply already in place when the row appears |
| `FadeOut(VGroup(l1, l2, l3), shift=UP*0.22)` — act 1 "clearing upward" | **deleted**, same hard cut |
| `FadeIn(cards symbols, shift=DOWN*0.12)` | plain `FadeIn`, 0.25 s |
| `FadeIn(rest3)` (the caption minus its first glyph) | `FadeIn(data3)` — the whole caption, since nothing transforms into it any more; `k4`/`rest3` are gone from the source |
| `FadeIn(c[1], shift=UP*0.12)` ×3, 0.60 s | plain `FadeIn`, 0.25 s |
| `FadeIn(res, shift=UP*0.12)`, 0.85 s | plain `FadeIn`, 0.25 s |
| `FadeIn(ax.group, shift=UP*0.10)`, 0.80 s | plain `FadeIn`, 0.45 s |
| `FadeIn(caveat)` 0.60 s at 11.15, `FadeIn(rescap)` 0.55 s at 11.90 | 0.25 s at 11.30 / 12.05 |

Kept, because they are **graphics**: `FadeIn(mark, shift=DOWN*0.25)` — the green result marker
dropping onto the bar — and PL d's fill lighting up (`bands["d"].animate.set_fill`).

⚠ manim's Cairo `Scene.remove` uses `extract_families=False`, so it only drops the exact objects
that `play()` added; `self.remove(big4, l1, l2, l3)` names them individually for that reason.

## Beats (14.803 s as built; 13.5 s planned)

| t (s) | what |
|---|---|
| 0.15–0.40 | a large red **4** fades up |
| 0.95–4.05 | "placements ended in contact" / "although the shield had verified the trajectory" / "of 2.65 × 10¹¹ uniform placements around the robot", each holding ~1 s |
| 5.00 | **hard cut**: act 1 is removed; the three factor symbols, the two `×`, the F/D legend and the red-`4` caption fade up (0.25 s) |
| 6.10–7.75 | each factor lights up with its value and its Beta-quantile definition (0.25 s each, ~0.45 s apart) |
| 8.30–8.55 | `PFH_D ≤ 9.50 × 10⁻⁷ / h` |
| 9.30–9.75 | the s01 PL axis returns |
| 10.20–11.05 | the green marker drops onto the bar, PL d lights up |
| 11.30–11.55 | "necessary, not sufficient, for PL d" |
| 12.05–12.30 | "at 99.999 % confidence", then a 2.50 s hold |

Cue times are absolute and were matched against `video/build/audio/s14.wav` (RMS pause detection:
gaps at 0.66, 1.68, 2.16, 3.92, 4.24, 5.86, 8.84, 11.44 s of audio, which starts 0.15 s into the
shot). `build.py` stretches the shot to **14.803 s** to fit the 14.30 s narration; the beats are
absolute, so that lands as a longer final hold rather than a drift.

## The row and the numbers

Row: `human_set = conformal`, `set_kind = conditional_conformal`, `mask_ood = True`,
`n_trials = 4e9` — the row S4 reports ("only 4 critical failures, achieving a PFH_D of 9.50e-7 at
99.999 % confidence").

| on screen | CSV column | value |
|---|---|---|
| N_h = 9 × 10⁵, "cycle time 4 ms" | `N_h`, `t_cycle` | 900 000 = 3600 s / 4 ms |
| P̄(F) ≤ 1.30 × 10⁻², "434 of 59 154 windows" | `p_f_upper`, `k_F`, `n_windows` | 1.2994987e-2 |
| P̄(D\|F) ≤ 8.12 × 10⁻¹¹, "**4** of 2.65 × 10¹¹ placements" | `p_d_given_f_upper`, `k_D`, `equivalent_uniform_placements` | 8.1244418e-11 |
| PFH_D ≤ 9.50 × 10⁻⁷ /h | `pfh_d_upper` | 9.5019312e-7 |
| at 99.999 % confidence | `epsilon_d` = 1e-5 | ε_F = ε_{D\|F} = 1−√(1−ε_D) = 5.0000125e-6 |
| marker position | `pfh_d_upper` on the PL axis | inside PL d (`pl` column = `d`) |

Supporting numbers named in the on-screen formulas but not spelled out: `l_corr` = 4.3236,
`n_eff` = 6840.79, `k_eff` = 50.189.

### The two Beta-quantile checks (run at render time, asserted to 1e-6 relative)

```
beta.ppf(1 - 5.0000125e-6, k_eff + 1, n_eff - k_eff)
    with k_eff = 50.18938570, n_eff = 6840.79014281
  = 0.01299498661626217      vs CSV p_f_upper         0.01299498661626217   (rel. err 0.0)

beta.ppf(1 - 5.0000125e-6, k_D + 1, N_D - k_D)
    with k_D = 4, N_D = 264 552 660 054.928
  = 8.124441862867928e-11    vs CSV p_d_given_f_upper 8.124441845370997e-11 (rel. err 2.2e-9)

9e5 * 1.2994987e-2 * 8.1244418e-11
  = 9.501931194531194e-07    vs CSV pfh_d_upper       9.501931174067680e-07 (rel. err 2.2e-9)
```

The 2.2e-9 residual is the only discrepancy and it is the *definition* difference described above:
the CSV stores `p_w · p_d_given_f_w_upper` (Clopper-Pearson over the 4 × 10⁹ witness-region
trials, scaled by the closed-form volume ratio) while the shot's check evaluates the paper's plain
Clopper-Pearson over the `2.6455 × 10¹¹` equivalent uniform placements. Agreeing to nine
significant figures is what licenses showing the paper's simpler form.

`load_row` also asserts `ε_F = ε_{D|F} = 1 − √(1 − ε_D)` (eq. `conf_composition`) and that
`pl == "d"` with `1e-7 ≤ pfh_d_upper < 1e-6`.

## Verification (2026-09-22 re-render)

`ffprobe`: 1920×1080, 30/1, yuv420p, progressive, **444 frames, 14.800000 s** at the build's
stretched duration (within `build.py`'s 0.08 s tolerance of 14.803).
Fourteen frames extracted from the encoded file and inspected (t = 0.5 / 1.5 / 2.7 / 4.3 / 4.9 /
5.4 / 6.5 / 7.3 / 8.0 / 8.7 / 9.9 / 11.0 / 12.5 / 14.7 s), including both sides of the 5.00 s
hard cut. Safe area measured over **all 444 decoded frames**: max luminance above y = 40 px is
16/255 and below y = 928 px is 17/255, zero pixels > 90/255 in either band; drawn-ink extent
**rows 57 … 918** (the lowest element is the axis decade labels — the caveat line sits *above*
the axis for exactly this reason).
The lowest element is the axis decade labels; the caveat line sits *above* the axis for exactly
this reason.

## Typography notes

* Everything sans goes through `pla.txt` (Inter, rendered large and scaled down). `MathTex` is
  used for all formulas and all on-screen values.
* The third card's caption is a **single** `Text` (`"4 of 2.65 × 10¹¹ placements"`) with its first
  glyph recoloured `C_DANGER` — two separate `Text`s cannot be baseline-aligned (`aligned_edge=DOWN`
  lines up ink bottoms, so the digit rides up by the descender of "placements"). The recolouring
  is now the *only* link back to the opening `4`; it used to also be the `ReplacementTransform`
  target, which the motion rule removed.
* `tex_sci` / `uni_sci` format the CSV floats, so the mantissas (1.30, 8.12, 9.50, 2.65) come from
  the data rather than being typed in.

## Caveats

* The PL axis was reordered (PL a left → PL e right, PFH_D 10⁻⁴ … 10⁻⁸). The marker at
  9.50 × 10⁻⁷ lands **just inside the left edge of PL d** — 5 px inside the band at this axis
  width, because the bound clears the 10⁻⁶ boundary by only 5 %. That is the honest picture; the
  band's green outline and the fill that lights up carry the "inside" reading. `_pl_axis.py` was
  **not** modified, so s01 does not need re-rendering.
* Card 2's estimator is written with `k_eff` / `N_eff` (what the code actually does) while its
  sub-caption says "434 of 59 154 windows" (what the reader can hold on to). The autocorrelation
  correction `L_corr = 4.32` that relates them is not on screen — there is no room for it and the
  paper carries it.
* `scipy` is used at render time (already present in the `chmp-video` env; no new packages).

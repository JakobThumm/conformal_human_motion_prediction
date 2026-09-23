"""s14 -- the PFH_D bound, assembled factor by factor, landing on the s01 performance-level axis.

Narration: "Four of those placements ended in contact although the shield had verified the
trajectory.  Clopper-Pearson on both factors bounds the dangerous failure rate at nine point five
times ten to the minus seven per hour, with 99.999 percent confidence."

s13 (the dangerous-trial replay) was cut from the film, so this is the only shot in which the
viewer meets ``k_D = 4``: the shot opens on that number, spells out what it means, and then folds
it into the third factor of the bound.

Per ``SPEC.md`` "Motion discipline" every word, number and equation here enters on **opacity
only** (<= 0.25 s) and **nothing ever exits**: the opening ``4`` act is cleared with a hard cut
and the red ``4`` is simply already in place in the third factor's caption when the row appears
(it used to fly there with a ``ReplacementTransform``).  The only remaining motion is the green
result marker dropping onto the PL axis and PL d's fill lighting up -- both graphics.

Structure follows eq. (pfh_bound) of ``content/S3_methodology.tex`` **verbatim** -- three factors,
no witness-region term::

    PFH_D <= N_h * B^-1(1-eps_F;    k_eff+1, N_eff-k_eff)
                 * B^-1(1-eps_D|F;  k_D+1,   N_D-k_D)

Every number is read at render time from the repository's own simulation summary

    results/final/robot_shield_risk_volume/shield_risk_volume_results.csv

row (human_set = conformal, set_kind = conditional_conformal, mask_ood = True, n_trials = 4e9),
i.e. the row the paper reports in S4 ("a PFH_D of 9.50e-7 at 99.999 % confidence"):

    N_h              = 9e5          (= 3600 s / t_cycle, t_cycle = 4 ms)
    p_f_upper        = 1.2995e-2    B^-1(1-eps_F;   k_eff+1, N_eff-k_eff),
                                    k_F = 434 / 59 154 windows, L_corr = 4.324
                                    -> N_eff = 6840.79, k_eff = 50.189
    p_d_given_f_up   = 8.1244e-11   B^-1(1-eps_D|F; k_D+1,   N_D-k_D),
                                    k_D = 4, N_D = equivalent_uniform_placements = 2.6455e11
    product          = 9.5019e-7 = pfh_d_upper       (asserted against the CSV at render time)
    epsilon_d        = 1e-5         -> 99.999 % confidence; eq. (conf_composition) splits it
                                       symmetrically over the two Clopper-Pearson bounds.

``load_row`` does not trust the CSV's derived columns: it re-derives **both** Clopper-Pearson
factors with ``scipy.stats.beta.ppf`` and asserts they match the CSV, then asserts that the three
factors multiply to the CSV's own ``pfh_d_upper``.  If someone re-runs the simulation the shot
follows automatically, and if the bound stops reproducing the shot fails loudly rather than
drawing a number it cannot derive.

The closing caveat is the paper's own sentence at the end of S3: "A bounded PFH_D is necessary but
not sufficient for a certain PL."

Render::

    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s14_pfhd.py \
        --out video/build/shots/s14.mp4 --duration 13.5
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEO_DIR))
sys.path.insert(0, str(VIDEO_DIR / "shots"))

import common  # noqa: E402
import style  # noqa: E402
import _pl_axis as pla  # noqa: E402

SID = "s14"
CSV_PATH = common.REPO / "results/final/robot_shield_risk_volume/shield_risk_volume_results.csv"


def load_row() -> dict:
    from scipy.stats import beta

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    sel = [r for r in rows
           if r["human_set"] == "conformal" and r["set_kind"] == "conditional_conformal"
           and r["mask_ood"] == "True" and float(r["n_trials"]) == 4e9]
    if len(sel) != 1:
        raise SystemExit(f"expected exactly one matching row in {CSV_PATH}, got {len(sel)}")
    r = sel[0]
    out = dict(
        N_h=float(r["N_h"]),
        p_f=float(r["p_f_upper"]),
        p_d=float(r["p_d_given_f_upper"]),
        pfh=float(r["pfh_d_upper"]),
        eps_d=float(r["epsilon_d"]),
        eps_f=float(r["epsilon_f"]),
        eps_df=float(r["epsilon_d_given_f"]),
        k_F=int(r["k_F"]), n_windows=int(r["n_windows"]),
        l_corr=float(r["l_corr"]), n_eff=float(r["n_eff"]), k_eff=float(r["k_eff"]),
        k_D=int(r["k_D"]), N_D=float(r["equivalent_uniform_placements"]),
        t_cycle=float(r["t_cycle"]), pl=r["pl"],
    )

    # -- eq. (conf_composition): the budget is split symmetrically over the two bounds ---------
    eps_sym = 1.0 - math.sqrt(1.0 - out["eps_d"])
    assert abs(out["eps_f"] / eps_sym - 1.0) < 1e-12, (out["eps_f"], eps_sym)
    assert abs(out["eps_df"] / eps_sym - 1.0) < 1e-12, (out["eps_df"], eps_sym)

    # -- eq. (pfh_bound), factor 1: P(F) over the effective sample size ------------------------
    p_f_beta = float(beta.ppf(1.0 - out["eps_f"], out["k_eff"] + 1.0,
                              out["n_eff"] - out["k_eff"]))
    assert abs(p_f_beta / out["p_f"] - 1.0) < 1e-6, (p_f_beta, out["p_f"])

    # -- eq. (pfh_bound), factor 2: P(D|F) over N_D uniform placements -------------------------
    # The CSV stores this as p_w * p_d_given_f_w_upper (the simulation draws its trials from the
    # witness region W and converts); it is numerically the *same* number as the paper's plain
    # Clopper-Pearson bound over the equivalent uniform placements, which is what we assert and
    # what the shot puts on screen.
    p_d_beta = float(beta.ppf(1.0 - out["eps_df"], out["k_D"] + 1.0, out["N_D"] - out["k_D"]))
    assert abs(p_d_beta / out["p_d"] - 1.0) < 1e-6, (p_d_beta, out["p_d"])

    # -- the three factors ARE the bound -------------------------------------------------------
    prod = out["N_h"] * out["p_f"] * out["p_d"]
    assert abs(prod / out["pfh"] - 1.0) < 1e-6, (prod, out["pfh"])
    assert out["pl"] == "d" and 1e-7 <= out["pfh"] < 1e-6, (out["pl"], out["pfh"])
    return out


# --------------------------------------------------------------------------- number formatting
_SUP = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074"
                                    "\u2075\u2076\u2077\u2078\u2079")


def _mant_exp(x: float, nd: int = 2) -> tuple[str, int]:
    """Mantissa (fixed to ``nd`` decimals, trailing zeros kept) and decimal exponent."""
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(m, nd) >= 10.0:                      # 9.995 -> 1.00e+1
        m, e = m / 10.0, e + 1
    return f"{m:.{nd}f}", e


def tex_sci(x: float, nd: int = 2) -> str:
    """LaTeX scientific notation; the mantissa is dropped when it is exactly 1."""
    m, e = _mant_exp(x, nd)
    return (rf"10^{{{e}}}" if float(m) == 1.0 else rf"{m} \times 10^{{{e}}}")


def uni_sci(x: float, nd: int = 2) -> str:
    """Unicode scientific notation for Inter body text."""
    m, e = _mant_exp(x, nd)
    return (f"10{str(e).translate(_SUP)}" if float(m) == 1.0
            else f"{m} × 10{str(e).translate(_SUP)}")


# --------------------------------------------------------------------------- render
def build_scene(duration: float, R: dict):
    from manim import (Scene, VGroup, MathTex, FadeIn, UP, DOWN, LEFT, RIGHT, rate_functions)

    def grp(x) -> str:
        return f"{int(x):,}".replace(",", " ")

    conf = 100.0 * (1.0 - R["eps_d"])
    n_d_txt = uni_sci(R["N_D"])                                   # "2.65 × 10¹¹"

    class S14(pla.TimedScene, Scene):
        D = duration

        # ---- one factor of the product: symbol / value / estimator / plain-language data ----
        def card(self, x, sym, val, est, data, est_scale=0.42):
            s = MathTex(sym, color=style.FG).scale(0.78)
            s.move_to([x, 2.92, 0])
            v = MathTex(val, color=style.YELLOW).scale(0.88)
            v.move_to([x, 2.26, 0])
            e = MathTex(est, color=style.FG_MUTED).scale(est_scale)
            e.move_to([x, 1.66, 0])
            d = data if data is not None else VGroup()
            if data is not None:
                d.move_to([x, 1.22, 0])
            return VGroup(s, v, e, d)

        def construct(self):
            self.add(pla.kicker("Results · probability of a dangerous failure"))

            # ---------------------------------------------------------------- act 1: k_D = 4
            big4 = pla.txt(f"{R['k_D']}", px=170, color=style.C_DANGER, weight="SEMIBOLD")
            l1 = pla.txt("placements ended in contact", px=40, color=style.FG)
            l2 = pla.txt("although the shield had verified the trajectory", px=40, color=style.FG)
            l3 = pla.txt(f"of {n_d_txt} uniform placements around the robot",
                         px=27, color=style.FG_MUTED)
            col = VGroup(l1, l2).arrange(DOWN, buff=0.20, aligned_edge=LEFT)
            l3.next_to(col, DOWN, buff=0.42).align_to(col, LEFT)
            act1 = VGroup(big4, VGroup(col, l3)).arrange(RIGHT, buff=0.62)
            act1.move_to([0.0, 0.62, 0])
            big4.align_to(col, UP).shift(DOWN * 0.06)

            # ---------------------------------------------------------------- the three factors
            xs = [-4.85, 0.0, 4.85]

            # The third card's caption is ONE Text so that the k_D digit sits on the same
            # baseline as the rest; its first glyph is recoloured so the eye can link it back
            # to the big red "4" that act 1 just showed.  (It used to be the target of a
            # ReplacementTransform from that "4" -- SPEC.md "Motion discipline" forbids moving
            # a number across the frame, so the caption is now simply already in place when
            # the row appears.)
            data3 = pla.txt(f"{R['k_D']} of {n_d_txt} placements", px=22, color=style.FG_MUTED)
            data3[0].set_color(style.C_DANGER)

            c1 = self.card(
                xs[0], r"N_h", tex_sci(R["N_h"], nd=0),
                r"3600\,\mathrm{s}\,/\,t_{\mathrm{cycle}}",
                pla.txt(f"cycle time {R['t_cycle'] * 1e3:.0f} ms", px=22,
                        color=style.FG_MUTED))
            c2 = self.card(
                xs[1], r"\overline{P}(F)", r"\le " + tex_sci(R["p_f"]),
                r"\mathrm{B}^{-1}(1-\epsilon_F;\, k_{\mathrm{eff}}{+}1,"
                r"\, N_{\mathrm{eff}}{-}k_{\mathrm{eff}})",
                pla.txt(f"{grp(R['k_F'])} of {grp(R['n_windows'])} windows", px=22,
                        color=style.FG_MUTED))
            c3 = self.card(
                xs[2], r"\overline{P}(D \mid F)", r"\le " + tex_sci(R["p_d"]),
                r"\mathrm{B}^{-1}(1-\epsilon_{D|F};\, k_D{+}1,\, N_D{-}k_D)",
                data3)
            cards = [c1, c2, c3]
            dots = VGroup(*[MathTex(r"\times", color=style.FG_MUTED).scale(0.8)
                            .move_to([0.5 * (xs[i] + xs[i + 1]), 2.26, 0]) for i in range(2)])

            legend = pla.txt(
                "F  the truth leaves the conformal prediction set   ·   "
                "D  contact although the shield verified the trajectory",
                px=23, color=style.FG_MUTED)
            legend.move_to([0, 0.62, 0])

            # ---------------------------------------------------------------- the result
            res = MathTex(r"\mathrm{PFH_D}", r"\;\le\;", tex_sci(R["pfh"]),
                          r"\;/\,\mathrm{h}", color=style.FG).scale(1.05)
            res[2].set_color(style.C_TRUTH)
            res.move_to([0, -0.10, 0])
            rescap = pla.txt(f"at {conf:.3f} % confidence", px=26, color=style.FG_MUTED)
            rescap.next_to(res, DOWN, buff=0.24)

            ax = pla.PLAxis(center=(0.0, -2.14, 0.0), width=8.2, bar_h=0.40, label_px=24)
            mark = ax.marker(R["pfh"], color=style.C_TRUTH, size=0.17)
            caveat = pla.txt(f"necessary, not sufficient, for PL {R['pl']}",
                             px=27, color=style.FG)
            caveat.move_to([0, -1.18, 0])

            # ---- timeline ------------------------------------------------------------------
            # Cued against video/build/audio/s14.wav (audio starts 0.15 s into the shot):
            #   "Four"                       0.15 s
            #   "...ended in contact"        ~1.0 s
            #   "...verified the trajectory" ~4.5 s, sentence ends ~6.0 s
            #   "Clopper-Pearson..."         ~6.1 s
            #   "nine point five times..."   ~9.0 s
            #   "with 99.999 percent..."     ~11.8 s
            #
            # SPEC.md "Motion discipline": every word/number/equation here enters on opacity
            # alone, in <= 0.25 s, and nothing ever exits.  The only motion left in the shot is
            # the green marker dropping onto the PL axis and PL d's fill lighting up.
            self.at(0.15)
            self.pl(FadeIn(big4), run_time=0.25)
            self.at(0.95)
            self.pl(FadeIn(l1), run_time=0.25)
            self.at(2.20)
            self.pl(FadeIn(l2), run_time=0.25)
            self.at(3.80)
            self.pl(FadeIn(l3), run_time=0.25)

            # Hard cut: act 1 is removed, not animated away, and the "4" is simply already in
            # place inside the third factor's caption when the row appears.
            self.at(5.00)
            self.remove(big4, l1, l2, l3)
            self.pl(FadeIn(VGroup(*[c[0] for c in cards])), FadeIn(dots), FadeIn(data3),
                    FadeIn(legend), run_time=0.25)

            # c3's caption is already on screen (it carries the k_D = 4 the act just showed),
            # so only its value and Beta-quantile definition are left to light up.
            for t0, c in zip((6.10, 6.80, 7.50), cards):
                anims = [FadeIn(c[1]), FadeIn(c[2])]
                if c is not c3:
                    anims.append(FadeIn(c[3]))
                self.at(t0)
                self.pl(*anims, run_time=0.25)

            self.at(8.30)
            self.pl(FadeIn(res), run_time=0.25)
            self.at(9.30)
            self.pl(FadeIn(ax.group), run_time=0.45)
            self.at(10.20)
            self.pl(FadeIn(mark, shift=DOWN * 0.25),          # graphic: the marker lands
                    ax.bands["d"].animate.set_fill(style.GREEN, opacity=0.34),
                    run_time=0.85, rate_func=rate_functions.ease_out_cubic)
            self.at(11.30)
            self.pl(FadeIn(caveat), run_time=0.25)
            self.at(12.05)
            self.pl(FadeIn(rescap), run_time=0.25)
            self.finish()

    return S14


def main() -> None:
    args = common.shot_args("s14 PFH_D bound")
    R = load_row()

    from manim import config
    config.media_dir = f"/tmp/manim_{SID}"
    config.disable_caching = True
    config.output_file = SID
    config.verbosity = "ERROR"
    config.progress_bar = "none"
    style.manim_config(config, quality_fps=args.fps)
    config.pixel_width, config.pixel_height = args.width, args.height
    config.background_color = style.BG

    scene = build_scene(args.duration, R)()
    scene.render()
    common.conform(Path(scene.renderer.file_writer.movie_file_path), args.out, args.duration,
                   fps=args.fps, width=args.width, height=args.height)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

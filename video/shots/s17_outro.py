"""s17 -- closing card: three takeaways.  Calm, static, holds to the end.

Narration: "Vision-based perception, with a number you can certify against."

Numbers on screen
-----------------
* "7.6x smaller"  -- ``content/S4_experiments.tex``: "Our conformal prediction sets achieve a 7.6
  times smaller median volume than the sets from the ISO 13855 model" (median 0.090 m^3 vs
  0.686 m^3 in ``results/final/conformal_prediction_sets/coverage_stats_*.csv``).  NOTE: the
  abstract and S1 round this to "8 times"; the film's s10 narration says "seven point six", so the
  card uses 7.6x for internal consistency.
* "1 - eps = 99.99 %" -- S4 ("calibration confidence of 1-eps = 99.99 %").
* "PFH_D <= 9.5e-7 /h at 99.999 % confidence" -- S4 and
  ``results/final/robot_shield_risk_volume/shield_risk_volume_results.csv``
  (pfh_d_upper = 9.50193e-7, epsilon_d = 1e-5) -- the same row s14 assembles.
* "24 % fewer" -- S4 / ``results/final/full_pipeline/``: invalid pose buffers 16.57 % -> 12.63 %.
* URL -- ``content/A0_abstract.tex`` (non-anonymous branch):
  https://jakob-thumm.com/conformal_human_motion_prediction/
  The manuscript currently compiles with \\setboolean{anon}{true}, so the URL is OFF by default
  and the card simply ends on the three takeaways -- there is no anonymous stand-in line.  Pass
  ``--show-url`` (or set ``args=["--show-url"]`` on shot s17 in ``video/script.py``) for the
  camera-ready cut, which re-tightens the layout to make room for it.

Render::

    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s17_outro.py \
        --out video/build/shots/s17.mp4 --duration 6.0
"""
from __future__ import annotations

import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEO_DIR))
sys.path.insert(0, str(VIDEO_DIR / "shots"))

import common  # noqa: E402
import style  # noqa: E402
import _pl_axis as pla  # noqa: E402

SID = "s17"
TITLE = "Vision-Based Safe Human–Robot Collaboration"
URL = "jakob-thumm.com/conformal_human_motion_prediction"

TAKEAWAYS = [
    (style.C_OURS,  "7.6× smaller conformal prediction sets   ·   1−ε = 99.99 %"),
    (style.C_TRUTH, "PFH<sub>D</sub> ≤ 9.5 × 10<sup>−7</sup> per hour   ·   "
                    "99.999 % confidence"),
    (style.C_OOD,   "24 % fewer interrupted predictions from OOD handling"),
]


SHOW_URL = False   # set by main() from --show-url; anonymous review is the default


def build_scene(duration: float):
    from manim import Scene, VGroup, Dot, FadeIn, LEFT

    class S17(pla.TimedScene, Scene):
        D = duration

        def construct(self):
            self.add(pla.kicker("Summary"))

            # Without the URL the card is title + three rows, so the block is re-centred
            # (a little above optical centre, because the burned-in subtitle owns the
            # bottom band) and the rows breathe wider.  --show-url restores the tighter
            # spacing that leaves room for the link.
            title_y, row_y, row_dy = (2.35, 0.85, 0.85) if SHOW_URL else (2.00, 0.50, 0.95)

            title = pla.txt(TITLE, px=46, weight="MEDIUM", color=style.FG)
            title.move_to([0, title_y, 0])

            rows, y = [], row_y
            for colour, text in TAKEAWAYS:
                t = pla.mtxt(text, px=34, color=style.FG)
                t.move_to([-3.05, y, 0], aligned_edge=LEFT)
                d = Dot([-3.55, y, 0], radius=0.085, color=colour)
                rows.append(VGroup(d, t))
                y -= row_dy

            # SPEC.md "Motion discipline": the card is nothing but words, so every element
            # enters on opacity alone (<= 0.25 s) and nothing ever leaves.  The rows used to
            # slide in from the right (`shift=RIGHT*0.20`) and the title to drop in
            # (`shift=DOWN*0.15`) -- both are gone, and the time they cost is now hold.
            self.pl(FadeIn(title), run_time=0.25)
            for r in rows:
                self.at(self._t + 0.10)
                self.pl(FadeIn(r), run_time=0.22)
            if SHOW_URL:
                url = pla.txt(URL, px=30, color=style.FG_MUTED)
                url.move_to([0, -2.05, 0])
                self.at(self._t + 0.15)
                self.pl(FadeIn(url), run_time=0.22)
            self.finish(min_hold=1.5)

    return S17


def main() -> None:
    global SHOW_URL
    args = common.shot_args(
        "s17 closing card",
        extra=lambda p: p.add_argument(
            "--show-url", action="store_true",
            help="print the project URL (de-anonymises: the manuscript is in anon mode)"),
    )
    SHOW_URL = args.show_url

    from manim import config
    config.media_dir = f"/tmp/manim_{SID}"
    config.disable_caching = True
    config.output_file = SID
    config.verbosity = "ERROR"
    config.progress_bar = "none"
    style.manim_config(config, quality_fps=args.fps)
    config.pixel_width, config.pixel_height = args.width, args.height
    config.background_color = style.BG

    scene = build_scene(args.duration)()
    scene.render()
    common.conform(Path(scene.renderer.file_writer.movie_file_path), args.out, args.duration,
                   fps=args.fps, width=args.width, height=args.height)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

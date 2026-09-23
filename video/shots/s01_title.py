r"""s01 -- *failure vs dangerous failure*, then the 10^-6 / h target.

Narration: "Cameras drop frames.  Predictions go wrong.  That is a failure.  It turns dangerous
only if the robot then touches the person.  Standards cap dangerous failures at one per million
hours: one in a hundred and fourteen years of round-the-clock work."

The film's **title card has moved to s02**, which now opens the film, so this shot no longer
carries the title/subtitle at all: it starts directly on the idea and spends the ~2.7 s it used
to lose on the two explanations.  Per ``SPEC.md`` "Motion discipline", every word on screen here
enters on **opacity only** (<= 0.25 s) and **nothing ever exits**: the F/D beat is cleared with a
hard cut when the PL axis takes over.  Only the graphics move (the escape ring and the impact
burst growing, the robot being drawn, the dashed 10^-6 rule being drawn, the bands fading up).

The shot has two beats:

1. two side-by-side pictograms that define the two events of ``content/S3_methodology.tex``
   ("Probability of Dangerous Failures Per Hour"):

       F  prediction failure  --  exists j, k: p^j_k not in S^j_k   (the green human leaves the
                                  blue conformal prediction set)
       D  dangerous failure   --  a contact occurs although c_safe = 1  (the robot additionally
                                  touches the human)

   Since the sets over-approximate the occupancy, ``D => F``: the right pictogram is literally the
   left one plus the robot, which is how the implication is drawn;
2. the ISO 13849-1 PL axis (shared ``_pl_axis.PLAxis``, PL a left -> PL e right) with the
   10^-6 / h cap, translated into years at render time.

Numbers / claims on screen and where they come from
---------------------------------------------------
* PL bands .................. ``_pl_axis.PL_BANDS``, which mirrors the repo's own classifier
  ``generate_plots/conformal_results_common.py::PL_BANDS``; PL d = [1e-7, 1e-6) is stated in
  ``content/S1_introduction.tex``, together with "collaborative robot applications commonly target
  PL d as specified in ISO 10218-1:2021".
* 114 years ................. computed here, not hardcoded: 1 / 1e-6 h = 1e6 h, divided by
  24 h x 365.25 d = 8766 h per year -> 114.1 years.  The arithmetic is printed on screen.
* background still .......... one frame (n = 1300) of ``video/media/realworld_source.mp4``, the
  real deployment recording, cropped to the 2D-pose overlay panel and graded down.  It used to
  fade up to 40 % to carry the title; now it just sits at its 8.5 % body level for the whole
  shot -- static, no crossfade.

The film's title is **not** here any more (it is s02's job); the submission is anonymised
(``main.tex`` sets ``\setboolean{anon}{true}``) so no authors/venue appear anywhere in the film.

Prepare (writes the cached still; re-run only if the source recording changes)::

    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s01_title.py --prepare

Render::

    /home/thumm/miniconda3/envs/chmp-video/bin/python video/shots/s01_title.py \
        --out video/build/shots/s01.mp4 --duration 14.5
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEO_DIR))
sys.path.insert(0, str(VIDEO_DIR / "shots"))

import common  # noqa: E402
import style  # noqa: E402
import _pl_axis as pla  # noqa: E402

SID = "s01"
SOURCE = common.MEDIA / "realworld_source.mp4"
STILL = common.MEDIA / "s01_frame.png"

# The backdrop level for the whole shot.  Set to 0.0 for a pure-black card.
BACKDROP_OPACITY = 0.085

# ---------------------------------------------------------------- the target, in hours and years
PFHD_CAP = 1e-6                       # ISO 13849-1:2023, upper end of PL d (S1 of the manuscript)
HOURS_PER_YEAR = 24.0 * 365.25        # round-the-clock operation, Julian year
CAP_HOURS = 1.0 / PFHD_CAP            # 1e6 h between two dangerous failures
CAP_YEARS = CAP_HOURS / HOURS_PER_YEAR  # -> 114.08 years


def _grp(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


# --------------------------------------------------------------------------- prepare
def prepare() -> None:
    """Grab + grade the single deployment frame used as the title backdrop."""
    vf = (
        "select='eq(n\\,1300)',"
        "crop=648:364:13:988,"                       # the RViz "Pose 2D Overlay" panel
        "scale=1920:1080:flags=lanczos,"
        "gblur=sigma=7,"
        "eq=brightness=-0.20:saturation=0.32:contrast=0.88,"
        "vignette=PI/4"
    )
    cmd = [common.FFMPEG, "-y", "-v", "error", "-i", str(SOURCE), "-vf", vf,
           "-frames:v", "1", "-fps_mode", "passthrough", str(STILL)]
    subprocess.run(cmd, check=True)
    print(f"wrote {STILL} ({STILL.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- pictograms
# One little scene, drawn twice: green = the true human (C_TRUTH), blue dashed = our conformal
# prediction sets (C_OURS), grey = the robot (C_ROBOT), red = contact (C_DANGER).
HEAD = (0.00, 1.16)
JOINTS = dict(
    neck=(0.00, 0.95), pelvis=(0.00, 0.30),
    sh_l=(-0.26, 0.90), sh_r=(0.26, 0.90),
    el_l=(-0.44, 0.52), el_r=(0.54, 1.02),
    ha_l=(-0.50, 0.14), ha_r=(0.92, 1.20),          # the right hand is reaching out
    hip_l=(-0.19, 0.30), hip_r=(0.19, 0.30),
    kn_l=(-0.23, -0.22), kn_r=(0.25, -0.22),
    ft_l=(-0.25, -0.76), ft_r=(0.28, -0.76),
)
BONES = [("neck", "pelvis"), ("neck", "sh_l"), ("neck", "sh_r"),
         ("sh_l", "el_l"), ("el_l", "ha_l"), ("sh_r", "el_r"), ("el_r", "ha_r"),
         ("pelvis", "hip_l"), ("pelvis", "hip_r"),
         ("hip_l", "kn_l"), ("kn_l", "ft_l"), ("hip_r", "kn_r"), ("kn_r", "ft_r")]
# where the model *predicted* those joints -- identical except the reaching hand, which is the
# one that escapes its set (and, in the right-hand panel, the one the robot hits)
SETS = [(HEAD, 0.30), (JOINTS["pelvis"], 0.36), (JOINTS["ha_l"], 0.28),
        (JOINTS["ft_l"], 0.26), (JOINTS["ft_r"], 0.26), ((0.40, 0.70), 0.30)]
ESCAPED = JOINTS["ha_r"]
CONTACT = (0.99, 1.19)


def _p(xy, dx=0.0, dy=0.0):
    return [xy[0] + dx, xy[1] + dy, 0.0]


def build_pictogram(with_robot: bool):
    """Return (group, contact_point) for one panel, in local coordinates."""
    from manim import VGroup, Circle, Line, Dot, DashedVMobject, Rectangle

    g = VGroup()
    floor = Line(_p((-1.45, -0.80)), _p((2.15, -0.80)), stroke_width=1.6, color=style.GRID)
    g.add(floor)

    # our conformal prediction sets
    sets = VGroup(*[DashedVMobject(
        Circle(radius=r, stroke_width=2.2, color=style.C_OURS).move_to(_p(c)),
        num_dashes=26, dashed_ratio=0.55) for c, r in SETS])
    g.add(sets)

    # the true human
    body = VGroup(Circle(radius=0.15, stroke_width=4.5, color=style.C_TRUTH).move_to(_p(HEAD)))
    for a, b in BONES:
        body.add(Line(_p(JOINTS[a]), _p(JOINTS[b]), stroke_width=4.5, color=style.C_TRUTH))
    g.add(body)

    # the escaped joint: outside its blue set
    esc = VGroup(Circle(radius=0.15, stroke_width=3.4, color=style.C_DANGER).move_to(_p(ESCAPED)),
                 Dot(_p(ESCAPED), radius=0.055, color=style.C_DANGER))

    robot = VGroup()
    if with_robot:
        base = Rectangle(width=0.54, height=0.15, stroke_width=0,
                         fill_color=style.C_ROBOT, fill_opacity=0.85).move_to(_p((1.88, -0.72)))
        links = [((1.88, -0.68), (1.80, 0.45)), ((1.80, 0.45), (1.30, 1.06)),
                 ((1.30, 1.06), (1.06, 1.17))]
        robot.add(base)
        for i, (a, b) in enumerate(links):
            robot.add(Line(_p(a), _p(b), stroke_width=9 - 2 * i, color=style.C_ROBOT))
        for a, b in links:
            robot.add(Dot(_p(a), radius=0.065, color=style.C_ROBOT))

    # contact: an impact burst on the joint the robot reaches, kept graphically distinct from
    # the escape ring so both events stay readable in the same panel
    hit = VGroup(Dot(_p(CONTACT), radius=0.085, color=style.C_DANGER))
    for i in range(8):
        a = np.pi / 8 + i * np.pi / 4
        hit.add(Line(_p((CONTACT[0] + 0.23 * np.cos(a), CONTACT[1] + 0.23 * np.sin(a))),
                     _p((CONTACT[0] + 0.37 * np.cos(a), CONTACT[1] + 0.37 * np.sin(a))),
                     stroke_width=3.2, color=style.C_DANGER))
    return g, sets, esc, robot, hit


# --------------------------------------------------------------------------- render
def build_scene(duration: float):
    from manim import (Scene, VGroup, FadeIn, Create, GrowFromCenter, LaggedStart,
                       Line, MathTex, RIGHT, ImageMobject, RoundedRectangle,
                       rate_functions)

    CARD_W, CARD_H, CARD_Y, CARD_X = 6.32, 3.95, 0.28, 3.42
    DX, DY = -0.30, 0.04                      # pictogram offset inside its card

    class S01(pla.TimedScene, Scene):
        D = duration

        # -------------------------------------------------------------- helpers
        def card(self, cx: float, chip_text: str, chip_color: str, caption: str,
                 with_robot: bool):
            frame = RoundedRectangle(width=CARD_W, height=CARD_H, corner_radius=0.16,
                                     stroke_width=1.5, stroke_color=style.GRID,
                                     fill_color=style.BG_PANEL, fill_opacity=0.42)
            frame.move_to([cx, CARD_Y, 0])
            chip = pla.txt(chip_text, px=34, weight="MEDIUM", color=chip_color)
            chip.move_to([cx, CARD_Y + 1.60, 0])
            pic, sets, esc, robot, hit = build_pictogram(with_robot)
            for m in (pic, esc, robot, hit):
                m.shift([cx + DX, DY, 0])
            cap = pla.mtxt(caption, px=25, color=style.FG_MUTED)
            cap.move_to([cx, CARD_Y - 1.66, 0])
            return dict(frame=frame, chip=chip, pic=pic, sets=sets, esc=esc, robot=robot,
                        hit=hit, cap=cap)

        def construct(self):
            # ---------------------------------------------------------- backdrop (static)
            # A *graphic*, not a word: the graded deployment still.  It used to crossfade up to
            # 40 % to carry the title; the title is s02's now, so the backdrop simply sits at
            # its body level from frame 0.  Nothing about it animates.
            img = ImageMobject(str(STILL)) if STILL.exists() else None
            if img is not None:
                img.stretch_to_fit_width(style.WIDTH / 135.0)
                img.stretch_to_fit_height(style.HEIGHT / 135.0)
                img.set_opacity(BACKDROP_OPACITY)
                img.set_z_index(-10)
                self.add(img)

            self.add(pla.kicker("The requirement"))

            # ---------------------------------------------------------- beat 1: F vs D
            left = self.card(
                -CARD_X, "Failure", style.YELLOW,
                f"the <span foreground='{style.C_TRUTH}'>human</span> leaves our "
                f"<span foreground='{style.C_OURS}'>predicted set</span>", with_robot=False)
            right = self.card(
                +CARD_X, "Dangerous failure", style.C_DANGER,
                f"…and the <span foreground='{style.C_ROBOT}'>robot</span> "
                f"<span foreground='{style.C_DANGER}'>touches</span> the "
                f"<span foreground='{style.C_TRUTH}'>human</span>", with_robot=True)
            plus = MathTex("+", color=style.FG_MUTED).scale(1.15).move_to([0, CARD_Y, 0])
            implies = pla.mtxt(
                f"<span foreground='{style.C_DANGER}'>dangerous failure</span>"
                f"   =   <span foreground='{style.YELLOW}'>failure</span>   +   robot contact",
                px=30, color=style.FG)
            implies.move_to([0, -2.22, 0])

            # ---------------------------------------------------------- beat 2: the target
            # Lifted by AX_DY relative to the old cut: the title header that used to occupy the
            # top of the frame is gone, so the whole axis block recentres and the arithmetic
            # line gains clearance from the 928 px subtitle band.
            AX_DY = 0.35
            ax_title = pla.mtxt(
                f"<span foreground='{style.C_DANGER}'>dangerous failures</span> per hour"
                "   ·   PFH<sub>D</sub>", px=30, color=style.FG)
            ax_title.move_to([0, 1.74 + AX_DY, 0])
            iso = pla.txt("ISO 13849-1:2023 performance levels  ·  collaborative robots "
                          "target PL d (ISO 10218-1:2021)", px=22, color=style.FG_MUTED)
            iso.move_to([0, 1.30 + AX_DY, 0])

            ax = pla.PLAxis(center=(0.0, 0.28 + AX_DY, 0.0), width=10.6, bar_h=0.46)
            thr = ax.threshold(PFHD_CAP, height=0.80)
            wash = ax.region(ax.LO, PFHD_CAP, color=style.GREEN, opacity=0.11)
            req = pla.mtxt("required: fewer than 10<sup>−6</sup> per hour",
                           px=26, weight="MEDIUM", color=style.GREEN)
            req.move_to([0, 0.95 + AX_DY, 0]).shift(RIGHT * (req.width / 2 + 0.26))
            dirs = ax.direction_labels()

            rule = Line([-4.6, -1.22 + AX_DY, 0], [4.6, -1.22 + AX_DY, 0],
                        stroke_width=1.0, color=style.GRID)
            years = pla.mtxt(
                "that is one dangerous failure every "
                f"<span foreground='{style.GREEN}'>{CAP_YEARS:.0f} years</span>",
                px=38, color=style.FG)
            years.move_to([0, -1.72 + AX_DY, 0])
            arith = pla.txt(
                f"one in {_grp(CAP_HOURS)} hours   ·   one year of round-the-clock work = "
                f"{_grp(HOURS_PER_YEAR)} hours (24 h × 365.25 d)   ·   "
                f"{_grp(CAP_HOURS)} / {_grp(HOURS_PER_YEAR)} = {CAP_YEARS:.1f} years",
                px=21, color=style.FG_MUTED)
            arith.move_to([0, -2.38 + AX_DY, 0])

            # ---------------------------------------------------------- timeline
            # (absolute cue times, matched against video/build/audio/s01.wav, which starts
            #  0.15 s into the shot: "…that is a failure" 3.4 s · "touches the person" 6.9 s ·
            #  "one per million hours" 10.1 s · "…a hundred and fourteen years" 12.0 s)
            #
            # The title card used to own 0.00-2.70 s.  That time is now spent on the two
            # explanations: the *Failure* panel is complete at 2.1 s instead of 4.1 s (so it
            # reads for 1.3 s on its own before the robot arrives, where it used to get 0.15 s),
            # and the *Dangerous failure* panel is complete at 6.1 s instead of 7.0 s.  The
            # arithmetic line lands 0.2 s earlier, which lengthens the closing hold to 2.55 s.
            #
            # Words only ever fade up, in <= 0.25 s, and nothing ever fades out.

            # -- failure ---------------------------------------------------------------
            self.at(0.20)
            self.pl(FadeIn(left["frame"]), FadeIn(left["pic"]), run_time=0.55)
            self.at(1.05)
            self.pl(GrowFromCenter(left["esc"]), run_time=0.45)     # graphic
            self.pl(FadeIn(left["chip"]), run_time=0.22)
            self.at(1.85)
            self.pl(FadeIn(left["cap"]), run_time=0.22)

            # -- dangerous failure ------------------------------------------------------
            self.at(3.45)
            self.pl(FadeIn(plus), FadeIn(right["frame"]), FadeIn(right["pic"]),
                    FadeIn(right["esc"]), run_time=0.55)
            self.pl(Create(right["robot"]), run_time=0.80,          # graphic
                    rate_func=rate_functions.ease_out_sine)
            self.at(5.05)
            self.pl(GrowFromCenter(right["hit"]), run_time=0.45)    # graphic
            self.pl(FadeIn(right["chip"]), run_time=0.22)
            self.at(5.85)
            self.pl(FadeIn(right["cap"]), run_time=0.22)
            self.at(6.40)
            self.pl(FadeIn(implies), run_time=0.25)

            # -- the target -------------------------------------------------------------
            # Hard cut: beat 1 is *removed*, not faded out.
            self.at(7.20)
            beat1 = [left["frame"], left["chip"], left["pic"], left["esc"], left["cap"],
                     right["frame"], right["chip"], right["pic"], right["esc"],
                     right["robot"], right["hit"], right["cap"], plus, implies]
            self.remove(*beat1)
            self.pl(FadeIn(ax_title), FadeIn(iso), run_time=0.25)
            self.at(7.65)
            self.pl(LaggedStart(*[FadeIn(VGroup(ax.bands[n], ax.labels[n]))
                                  for n, _, _ in pla.PL_BANDS],
                                lag_ratio=0.20), run_time=0.95)
            self.at(8.85)
            self.pl(FadeIn(ax.ticks), FadeIn(dirs), run_time=0.25)

            self.at(9.45)
            self.pl(Create(thr), FadeIn(wash),                      # graphics
                    ax.bands["d"].animate.set_fill(style.GREEN, opacity=0.44),
                    run_time=0.70)
            self.pl(FadeIn(req), run_time=0.22)

            self.at(10.45)
            self.pl(FadeIn(rule), FadeIn(years), run_time=0.25)
            self.at(11.70)
            self.pl(FadeIn(arith), run_time=0.25)
            self.finish()

    return S01


def main() -> None:
    args = common.shot_args("s01 title card + failure vs dangerous failure + the 1e-6/h target")
    if not STILL.exists():
        prepare()

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
    if "--prepare" in sys.argv:
        prepare()
    else:
        main()

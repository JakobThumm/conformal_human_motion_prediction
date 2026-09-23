"""Storyboard + narration for the ICRA submission video.

Single source of truth.  `build.py` reads SHOTS, synthesises the narration, renders every shot to
exactly `duration` seconds, and concatenates.  Editing a line here changes the film.

Shot contract (every renderer in `video/shots/` must honour it):

    <interp> video/shots/<module>.py --out OUT.mp4 --duration SECONDS [--fps 30] [--width 1920]
                                     [--height 1080]

  * writes a silent H.264 mp4 of EXACTLY `duration` seconds at 1920x1080, fps 30, yuv420p
  * deterministic: same inputs -> same frames
  * `interp` is "video" (conda env chmp-video: manim/matplotlib/cv2) or
    "repo" (the project .venv: jax + models + datasets)
  * keeps the bottom ~14 % of the frame free of essential content (burned-in subtitles live there)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Shot:
    sid: str                      # stable id, also the build filename
    module: str                   # video/shots/<module>.py
    duration: float               # planned seconds (build may stretch to fit narration)
    narration: str                # spoken text; "" = silent shot
    interp: str = "video"         # "video" | "repo"
    args: list = field(default_factory=list)   # extra argv for the renderer
    part: str = ""                # hook | method | results | outro
    subtitle: str = ""            # override subtitle text (default: narration)
    enabled: bool = True          # False = kept on disk, cut from the film


SHOTS: list[Shot] = [
    Shot(
        "s02", "s02_problem", 8.5, part="hook",
        narration=(
            "Our goal: from camera input alone, guarantee the robot is always at a "
            "complete stop before it could touch the human."
        ),
    ),
    Shot(
        "s01", "s01_title", 14.5, part="hook",
        narration=(
            "Cameras drop frames. Predictions go wrong. That is a failure. It turns "
            "dangerous only if the robot then touches the person. Standards cap "
            "dangerous failures at one per million hours: one in a hundred and fourteen "
            "years of round-the-clock work."
        ),
    ),
    Shot(
        "s03", "s03_pose2d", 7.3, part="method", interp="repo",
        narration=(
            "It starts with two views. An adapted YOLO twenty-six returns, for every "
            "joint, not a pixel, but a covariance."
        ),
    ),
    Shot(
        "s04", "s04_triangulate", 9.0, part="method", interp="repo",
        narration=(
            "Uncertainty-aware triangulation lifts pose and covariance into three "
            "dimensions. Every joint arrives as an ellipsoid."
        ),
    ),
    Shot(
        "s05", "s05_motion", 11.5, part="method",
        narration=(
            "A transformer reads two seconds of this uncertain history, and predicts "
            "the next four hundred milliseconds, with covariances."
        ),
    ),
    Shot(
        "s06", "s06_conformal", 25.0, part="method",
        narration=(
            "But a learned covariance is a guess, not a guarantee. So we calibrate: on "
            "held-out data we measure how far the truth lands, in units of predicted "
            "sigma. Its 99.99th percentile scales every sphere into a conformal "
            "prediction set: a ball that contains the true joint with 99.99 percent "
            "probability."
        ),
    ),
    Shot(
        "s08", "s08_shield", 14.7, part="method",
        narration=(
            "The spheres become capsules: the human's reachable occupancy. SARA shield "
            "intersects them with the robot's. No intersection, go. Otherwise, "
            "fail-safe stop."
        ),
    ),
    Shot(
        "s07", "s07_ood", 12.0, part="method",
        narration=(
            "Conformal guarantees assume tomorrow looks like yesterday. A "
            "sketched-Lanczos monitor watches the networks' gradients, and out of distribution, "
            "we reuse the previous prediction instead of guessing."
        ),
    ),
    Shot(
        "s09", "s09_vs_iso", 9.0, part="results",
        narration=(
            "Orange is the constant-velocity model of ISO thirteen eight fifty-five. "
            "Blue is ours: the same confidence, in seven point six times less space."
        ),
    ),
    Shot(
        "s10", "s10_volume_chart", 8.2, part="results", enabled=False,
        narration=(
            "Across the test set our sets are seven point six times smaller, and the "
            "truth escapes them one point six times in ten thousand."
        ),
    ),
    Shot(
        "s11", "s11_sim_setup", 12.5, part="results",
        narration=(
            "Smaller is only worth having if it stays safe. So we take every prediction "
            "failure and drop it around the robot: random place in a ten-meter circle, "
            "random orientation, random phase of a pick-and-place trajectory."
        ),
    ),
    Shot(
        "s12", "s12_sim_prune", 9.2, part="results",
        narration=(
            "Almost none of those placements can reach the robot; two geometric filters "
            "prove it. In total we check two hundred sixty-five billion placements on "
            "the GPU."
        ),
    ),
    Shot(
        "s13", "s13_sim_danger", 10.4, part="results", enabled=False,
        narration=(
            "Four come back dangerous. The shield verified its trajectory, but the true "
            "occupancy escaped the prediction and touched the robot. This is one of the "
            "four, replayed exactly."
        ),
    ),
    Shot(
        "s14", "s14_pfhd", 13.5, part="results",
        narration=(
            "Four of those placements ended in contact although the shield had verified "
            "the trajectory. Clopper-Pearson on both factors bounds the dangerous "
            "failure rate at nine point five times ten to the minus seven per hour, "
            "with 99.999 percent confidence."
        ),
    ),
    Shot(
        "s15", "s15_ood_results", 7.0, part="results", enabled=False,
        narration=(
            "The fallback cuts invalid predictions by twenty-four percent, and the "
            "pipeline runs in sixty-six milliseconds."
        ),
    ),
    Shot(
        "s16", "s16_realworld", 21.0, part="results",
        narration=(
            "On hardware: a Franka arm, a RealSense camera, the same pipeline at "
            "twenty-five hertz. Blue is the human's conformal occupancy over the "
            "robot's stopping horizon. When it meets the robot's reachable set, the "
            "shield brakes. In every tested instance the robot stood still long before "
            "the operator could reach it."
        ),
    ),
    Shot(
        "s17", "s17_outro", 5.2, part="outro",
        narration=(
            "Vision-based perception, with a number you can certify against."
        ),
    ),
]


def active() -> list[Shot]:
    """The shots that actually make the cut, in film order."""
    return [s for s in SHOTS if s.enabled]


def total_planned() -> float:
    return sum(s.duration for s in active())


if __name__ == "__main__":
    for s in active():
        words = len(s.narration.split())
        print(f"{s.sid}  {s.part:8s} {s.duration:5.1f}s  {words:3d} words  "
              f"({words / max(s.duration, 1e-9):.2f} w/s)  {s.module}")
    print(f"total planned: {total_planned():.1f} s")

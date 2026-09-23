# ICRA submission video

A ~3-minute film about *Vision-Based Safe Human-Robot Collaboration*: one third methodology,
two thirds results, every number and every frame traceable to this repository.

Deliverables (written by `build.py assemble`):

| file | what |
|---|---|
| `out/icra_video.mp4` | the submission master — H.264, 1920×1080, 30 fps progressive, ≤ 180 s, ≤ 20 MB, narrated, **no** burned-in subtitles |
| `out/icra_video_subtitled.mp4` | same film with burned-in subtitles |
| `out/narration.srt`, `out/narration.vtt` | sidecar subtitles for the un-burned master |
| `out/narration_script.md` | the spoken script with timecodes |

## Build

```bash
cd /home/thumm/code/conformal_human_motion_prediction
/home/thumm/miniconda3/envs/chmp-video/bin/python video/build.py all
```

Stages (each runnable on its own):

| stage | what it does |
|---|---|
| `fetch` | copies source media into `video/media/`, downloads the Piper voice |
| `narrate` | synthesises every narration line, fits shot durations to the voice, writes `build/timeline.json` and prints the time budget (must stay ≤ 180 s) |
| `shots [--only s06,s09] [--force]` | runs each renderer in `video/shots/` to exactly its timeline duration |
| `assemble` | concatenates, mixes narration + ducked lab ambience, burns subtitles, two-pass encodes to the 20 MB target |
| `check` | re-verifies the ICRA constraints on everything in `out/` |

`build.py shots --placeholders` stubs every shot, which is how the assembly path is smoke-tested
without rendering anything.

## Layout

```
video/
├── script.py       storyboard + narration -- the single source of truth for what is said when
├── style.py        palette, type, canvas (imported by every renderer)
├── common.py       shot CLI, exact-duration frame writer, easing helpers
├── tts.py          Piper narration, SRT/VTT/ASS subtitle generation
├── build.py        orchestrator
├── SPEC.md         the binding contract every shot renderer follows
├── shots/          one renderer per shot (s01 … s17) + per-shot notes
├── media/          every external asset the film uses, copied in (self-contained)
├── assets/voices/  Piper voice model (downloaded, ~120 MB, git-ignored)
├── build/          intermediates: per-shot mp4s, narration wavs, timeline.json
└── out/            the deliverables
```

## Environments

* **video env** — `/home/thumm/miniconda3/envs/chmp-video` (conda-forge): manim 0.20, ffmpeg 9,
  matplotlib, opencv, piper-tts. Created with
  `conda create -n chmp-video -c conda-forge python=3.11 manim ffmpeg` followed by
  `pip install matplotlib scipy opencv-python-headless pandas cloudpickle piper-tts`
  (manim cannot be pip-installed here: `pycairo`/`manimpango` need system cairo/pango headers).
* **repo env** — the project `.venv`, for shots that run real model inference (jax + the H36M /
  RGB-D datasets). Those shots cache their output into `video/media/` so the film rebuilds without
  a GPU.

## What is deliberately not in git

The repository carries the *pipeline*, not the *renders*. Missing, and how to get each back:

| not tracked | how to restore |
|---|---|
| `out/*.mp4` — the film itself | `build.py all` |
| `build/` — per-shot renders, narration wavs, `timeline.json` | `build.py narrate && build.py shots` |
| `assets/voices/` — the Piper voice (~120 MB) | `build.py fetch` (downloads it) |
| `media/*.npz`, `media/*.png` — per-shot `--prepare` caches | the commands in the next section |
| `media/realworld_source.mp4` — **the lab recording** | copy the original capture back to the path named in `script.py::SOURCE_MEDIA`, then `build.py fetch` |

Only the last one is irreplaceable: everything else is a function of the code plus the
experiment outputs under `results/`. The subtitles, the timed narration script and `summary.txt`
in `out/` *are* tracked, since they are the human-readable record of what the film says.

## Regenerating the cached data (only needed after a model/result changes)

Shots that touch models, datasets or the 500 MB result pickles do their work once in a `--prepare`
step and commit a small `.npz`/`.png` into `video/media/`. The render path then runs offline. The
full set, in the order the film uses them:

```bash
cd /home/thumm/code/conformal_human_motion_prediction
V=/home/thumm/miniconda3/envs/chmp-video/bin/python
X="XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python"

$V     video/shots/s01_title.py       --prepare     # graded still from the deployment capture
env $X video/shots/s03_pose2d.py      --prepare     # YOLO26 2-D poses + per-keypoint covariance
env $X video/shots/s04_triangulate.py --prepare     # + uncertainty-aware triangulation
JAX_PLATFORMS=cpu .venv/bin/python video/shots/_human3d_prepare.py   # s05 + s09 windows
.venv/bin/python video/shots/s06_conformal.py      --prepare         # non-conformity histogram, alpha
.venv/bin/python video/shots/s07_ood.py            --prepare         # ID/OOD SLU scores + episode
.venv/bin/python video/shots/s10_volume_chart.py   --prepare         # 7.7 M set volumes
env $X video/shots/s1x_shield_data.py                                # s11/s12/s13 shield geometry
$V     video/shots/s15_ood_results.py --prepare                      # ablation + runtime tables
```

Each shot's `*_notes.md` in `video/shots/` records what it reads, what it asserts, and every
caveat its author found (they are worth reading before changing a number on screen).

## Editing the film

Change a narration line or a shot length in `script.py`, then

```bash
python video/build.py narrate --force     # re-synthesise, re-check the 180 s budget
python video/build.py shots               # re-render only the shots whose duration moved
python video/build.py assemble
```

Shot durations are *derived*: the narration is synthesised first and each shot is stretched to at
least `voice + 0.5 s`. Renderers therefore never hardcode a length — they take `--duration`.

## Narration

Offline neural TTS (Piper, `en_US-ryan-high`), so the build is deterministic and needs no network
after `fetch`. To record a human voice-over instead, drop 48 kHz wavs named `s01.wav … s17.wav`
into `build/audio/` and run `narrate` without `--force` (existing wavs are kept and only measured).

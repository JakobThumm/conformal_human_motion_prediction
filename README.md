# Project website — Vision-Based Safe Human-Robot Collaboration with Uncertainty Guarantees

This is the `gh-pages` branch. It hosts the [GitHub Pages](https://pages.github.com/) project
website for the paper, served at:

> https://jakobthumm.github.io/conformal_human_motion_prediction/

The page is a static site adapted from the
[Nerfies](https://github.com/nerfies/nerfies.github.io) project-page template (Bulma CSS).

## Editing the text

**Every word on the page lives in one file: [`index.html`](index.html).** It is plain HTML with no
build step and no templating — edit it, reload the browser, done. Each section is introduced by a
banner comment, so `grep -n '<!-- =====' index.html` gives you a table of contents. At the time of
writing:

| line | section | what is in it |
|---|---|---|
| 33 | `HERO` | title, author list and affiliations, the four link buttons |
| 103 | `FILM` | the YouTube embed and its one-line caption |
| 122 | `ABSTRACT` | the abstract, reproduced from `content/A0_abstract.tex` |
| 158 | `HIGHLIGHT STATS` | the four big numbers under the abstract |
| 190 | `THE REQUIREMENT` | what a dangerous failure is, and the PL d target |
| 222 | `METHOD / PIPELINE` | the overview figure, its caption, and the five numbered steps |
| 376 | `RESULTS` | setup paragraph, H1/H2/H3, all four tables, the scope caveat |
| 666 | `REAL-WORLD DEMO` | the Franka / RealSense paragraph |
| 696 | `BIBTEX` | the citation block |
| 709 | `FOOTER` | funding and template attribution |

Two conventions worth keeping as you edit:

- `&nbsp;` between a number and its unit (`7.6&nbsp;m`, `PL&nbsp;d`) stops them splitting across a
  line break; `class="nb"` does the same for a whole scientific-notation number.
- Styling lives in [`static/css/index.css`](static/css/index.css), never inline. If something needs
  a new look, add a class there.

Check your edits before committing:

```bash
python -m http.server 8000    # then open http://localhost:8000
```

## Structure

```
index.html                           # the page — all text lives here
static/css/index.css                 # custom styles (blue accent)
static/js/index.js                   # smooth scroll + lazy play/pause for the inline clips
static/images/pipeline_overview.png  # methodological overview (Fig. 1 of the paper)
static/images/film_poster.jpg        # social-preview image (og:image) only
static/videos/clip_*.mp4             # ten silent scenes cut from the film (960x540, 2.3 MB total)
static/videos/film.vtt               # caption track for the film — see below
tools/build_media.sh                 # regenerates everything under static/videos and static/images
.nojekyll                            # disable Jekyll so static/ is served verbatim
```

## The video

The full three-minute film is **not hosted here**. The page embeds
<https://www.youtube.com/watch?v=qN9ZHieRR7w>; to swap it, change the two occurrences of that ID in
`index.html` — the hero button and the `<iframe>` in the `FILM` section.

`static/videos/film.vtt` is kept as the canonical caption file even though the page does not
reference it. Upload it in **YouTube Studio → Subtitles → Add language → Upload file** so the
YouTube captions match the narration script exactly.

The ten short clips *are* self-hosted, because they are silent 10–20 s loops that play inline
alongside the text. Ten YouTube players would cost far more than the 2.3 MB they take up.

## Regenerating the media

Every video and image on the page is derived from the main repository — nothing here is a one-off
export. To rebuild after re-cutting the film:

```bash
./tools/build_media.sh [path-to-conformal_human_motion_prediction]   # default: ~/code/…
```

It reads `video/out/ICRA_2027.mp4` and `video/out/ICRA_2027.vtt` from the `supplementary-video`
branch of the main repo, plus `Vision-Based-Safe-Human-Robot-Collaboration/figures/`.

The clip cut points are hard-coded in that script, each annotated with the shot it corresponds to.
They were read off the **dubbed** master frame by frame, *not* taken from
`video/build/timeline.json` — the hand-dubbed cut re-times the shots, so the two disagree by up to
five seconds in the results half. If the film is re-cut, re-check them:

```bash
for t in $(seq 100 2 178); do
  ffmpeg -loglevel error -ss $t -i ICRA_2027.mp4 -frames:v 1 -vf scale=400:-1 f_$t.png
done
```

## Where the numbers come from

The page tracks the manuscript as of 2026-09-23: the abstract is reproduced verbatim, and Table I,
Table II, the *N*<sub>req</sub> ablation, the settings line and the AUROC figures all follow
`content/S4_experiments.tex` and `figures/result_tables/`. Headline numbers are 7.6× volume
reduction, PFH<sub>D</sub> ≤ 9.50 × 10⁻⁷ /h at 99.999 % confidence, and 24 % fewer interrupted
predictions — the same numbers the film shows.

Two things on the page are *not* `\input` by the manuscript and were taken from elsewhere:

- the **per-stage runtime table** comes from `figures/result_tables/runtime.tex`, which no section
  includes; its numbers appear in the paper only as prose in Sec. IV-D. (The supplementary has a
  different runtime table, measured on an RTX 4070 Super.)
- the film states an **AUROC of 0.992** for the pose monitor where Sec. IV-D now says 0.9976. The
  page follows the paper.

## Enabling GitHub Pages

Repo **Settings → Pages → Build and deployment**: set *Source* to **Deploy from a branch**,
branch **`gh-pages`**, folder **`/ (root)`**.

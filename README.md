# Project website — Vision-Based Safe Human-Robot Collaboration with Uncertainty Guarantees

This is the `gh-pages` branch. It hosts the [GitHub Pages](https://pages.github.com/) project
website for the paper, served at:

> https://jakobthumm.github.io/conformal_human_motion_prediction/

The page is a static site adapted from the
[Nerfies](https://github.com/nerfies/nerfies.github.io) project-page template (Bulma CSS).

## Structure

```
index.html                        # the page
static/css/index.css              # custom styles (blue accent)
static/js/index.js                # smooth-scroll helper
static/images/pipeline_overview.png  # methodological overview (Fig. 1)
.nojekyll                         # disable Jekyll so static/ is served verbatim
```

## Local preview

```bash
python -m http.server 8000    # then open http://localhost:8000
```

## Enabling GitHub Pages

Repo **Settings → Pages → Build and deployment**: set *Source* to **Deploy from a branch**,
branch **`gh-pages`**, folder **`/ (root)`**.

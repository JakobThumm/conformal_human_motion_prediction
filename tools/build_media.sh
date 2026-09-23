#!/usr/bin/env bash
# Regenerate every media asset on the project page from the sources in the main repo.
#
#   ./tools/build_media.sh [path-to-conformal_human_motion_prediction]
#
# Inputs (all from the `supplementary-video` branch of the main repo, plus the manuscript):
#   video/out/ICRA_2027.mp4                              the supplementary film (1920x1080, 30 fps)
#   video/out/ICRA_2027.vtt                              its subtitle track
#   Vision-Based-Safe-Human-Robot-Collaboration/figures/ paper figures
#
# Outputs: static/videos/*.mp4, static/videos/film.vtt, static/images/*
set -euo pipefail

REPO="${1:-$HOME/code/conformal_human_motion_prediction}"
FILM="$REPO/video/out/ICRA_2027.mp4"
VTT="$REPO/video/out/ICRA_2027.vtt"
FIGS="$REPO/Vision-Based-Safe-Human-Robot-Collaboration/figures"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HERE/static/videos"
IMG="$HERE/static/images"

[ -f "$FILM" ] || { echo "missing $FILM" >&2; exit 1; }
mkdir -p "$OUT" "$IMG"

# --- the full film is NOT hosted here; the page embeds https://youtu.be/qN9ZHieRR7w ---
# film.vtt is kept only so there is one canonical caption file to upload to YouTube
# (Studio -> Subtitles -> Upload file). It is not referenced by index.html.
echo "==> film.vtt"
cp "$VTT" "$OUT/film.vtt"

echo "==> film_poster.jpg   (og:image / social preview only)"
ffmpeg -y -loglevel error -ss 158 -i "$FILM" -frames:v 1 -vf "scale=1280:-1" -q:v 4 "$IMG/film_poster.jpg"

# --- inline scenes: silent, looping, small. Cut points come from ---------------------
# --- video/out/narration_script.md (timings measured against the dubbed audio). ------
# name            start   end
clip() {
  local name="$1" start="$2" end="$3"
  echo "==> clip_$name.mp4  ($start -> $end)"
  ffmpeg -y -loglevel error -ss "$start" -to "$end" -i "$FILM" \
    -an -vf "scale=960:540:flags=lanczos,fps=25" \
    -c:v libx264 -profile:v high -preset veryslow -crf 30 -pix_fmt yuv420p \
    -movflags +faststart "$OUT/clip_$name.mp4"
}

# Cut points were read off the dubbed film frame-by-frame (`ffmpeg -ss <t> -frames:v 1`),
# not from video/build/timeline.json -- the hand-dubbed master re-times the shots.
clip failure     8.7  22.8   # s01  what makes a failure dangerous; the 1-in-a-million requirement
clip perception 23.3  38.9   # s03  2-D pose + covariance, s04 uncertainty-aware triangulation
clip motion     39.4  50.9   # s05  DCT pose transformer, heteroscedastic covariances
clip conformal  51.3  75.2   # s06  non-conformity score, calibration, the guaranteed sphere
clip shield     75.6  90.9   # s08  capsules -> reachable occupancy -> SARA shield verification
clip ood        91.3 102.8   # s07  sketched-Lanczos monitors + Alg. 1 buffer fallback
clip iso       103.2 115.3   # s09  our sets vs the ISO 13855 constant-velocity model
clip sim       116.6 137.2   # s11 + s12  the 2.65e11-placement certification simulation
clip pfhd      137.6 155.2   # s14  Clopper-Pearson bound on PFH_D
clip realworld 155.7 177.0   # s16  Franka + RealSense deployment under SARA shield

# --- figures ------------------------------------------------------------------------
echo "==> pipeline_overview.png"
python3 - "$FIGS/overview_full_2.png" "$IMG/pipeline_overview.png" <<'PY'
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
im = im.resize((1800, round(1800 * im.height / im.width)), Image.LANCZOS)
im.save(dst, optimize=True)
print(f"   {im.size[0]}x{im.size[1]}")
PY

du -sh "$OUT" "$IMG"

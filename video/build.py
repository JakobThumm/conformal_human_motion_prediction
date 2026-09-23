"""Reproducible build of the ICRA submission video.

    python video/build.py all                 # narration -> shots -> assemble -> encode
    python video/build.py narrate             # (re)synthesise narration, print the timing budget
    python video/build.py shots --only s06,s09
    python video/build.py assemble            # mux + subtitles + size-targeted encode
    python video/build.py check               # verify the ICRA constraints on the deliverables

Run it with any python >= 3.10; it shells out to the two project interpreters:
  * VIDEO_PY = conda env `chmp-video` (manim, matplotlib, cv2, piper)   -- shots with interp=video
  * REPO_PY  = the project .venv (jax, models, datasets)                -- shots with interp=repo
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent
REPO = VIDEO_DIR.parent
sys.path.insert(0, str(VIDEO_DIR))

import script as storyboard  # noqa: E402
import tts  # noqa: E402

VIDEO_PY = Path("/home/thumm/miniconda3/envs/chmp-video/bin/python")
REPO_PY = REPO / ".venv/bin/python"
FFMPEG = Path("/home/thumm/miniconda3/envs/chmp-video/bin/ffmpeg")
FFPROBE = Path("/home/thumm/miniconda3/envs/chmp-video/bin/ffprobe")
if not FFMPEG.exists():
    FFMPEG, FFPROBE = Path("ffmpeg"), Path("ffprobe")

BUILD = VIDEO_DIR / "build"
SHOTS_DIR = BUILD / "shots"
AUDIO_DIR = BUILD / "audio"
OUT = VIDEO_DIR / "out"
MEDIA = VIDEO_DIR / "media"

TIMELINE_JSON = BUILD / "timeline.json"

# ICRA submission constraints
MAX_SECONDS = 180.0
# Read the 20 MB limit the strict way (20 x 10^6 bytes) and leave ~8 % headroom, so
# the file is comfortably under whichever convention the submission system uses.
MAX_BYTES = 20_000_000
MIN_HEIGHT = 480
MIN_FPS = 20

TAIL_PAD = 0.35      # silence appended after each narration line
LEAD_PAD = 0.15      # silence before it


# --------------------------------------------------------------------------- narration

def narrate(force: bool = False) -> list[dict]:
    """Synthesise every line, fit shot durations around it, write build/timeline.json."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    timeline, t0 = [], 0.0
    for sh in storyboard.active():
        wav = AUDIO_DIR / f"{sh.sid}.wav"
        adur = 0.0
        if sh.narration:
            if force or not wav.exists():
                tts.synthesize(sh.narration, wav)
            adur = tts.wav_duration(wav)
        need = adur + LEAD_PAD + TAIL_PAD
        dur = round(max(sh.duration, need), 3)
        timeline.append({
            "sid": sh.sid, "module": sh.module, "part": sh.part, "interp": sh.interp,
            "args": sh.args, "planned": sh.duration, "duration": dur,
            "audio_duration": adur, "audio_start": LEAD_PAD if adur else 0.0,
            "start": round(t0, 3), "text": sh.subtitle or sh.narration,
        })
        t0 += dur
    BUILD.mkdir(parents=True, exist_ok=True)
    tts.dump_timeline(timeline, TIMELINE_JSON)
    report(timeline)
    return timeline


def report(timeline: list[dict]) -> None:
    total = sum(i["duration"] for i in timeline)
    parts: dict[str, float] = {}
    print(f"{'id':5s} {'part':8s} {'plan':>6s} {'voice':>6s} {'final':>6s}  slack")
    for i in timeline:
        parts[i["part"]] = parts.get(i["part"], 0.0) + i["duration"]
        slack = i["duration"] - (i["audio_duration"] + LEAD_PAD + TAIL_PAD)
        flag = "  <-- stretched" if i["duration"] > i["planned"] + 1e-6 else ""
        print(f"{i['sid']:5s} {i['part']:8s} {i['planned']:6.1f} {i['audio_duration']:6.1f} "
              f"{i['duration']:6.1f} {slack:6.1f}{flag}")
    print("-" * 46)
    for k, v in parts.items():
        print(f"{k:12s} {v:6.1f} s   ({v / total * 100:4.1f} %)")
    meth = parts.get("method", 0.0)
    res = parts.get("results", 0.0)
    print(f"method:results = 1 : {res / max(meth, 1e-9):.2f}   (target 1 : 2)")
    print(f"TOTAL {total:.1f} s   limit {MAX_SECONDS:.0f} s   "
          f"{'OK' if total <= MAX_SECONDS else 'OVER BUDGET'}")


def load_timeline() -> list[dict]:
    if not TIMELINE_JSON.exists():
        return narrate()
    return json.loads(TIMELINE_JSON.read_text())


# --------------------------------------------------------------------------- shots

def render_shots(only: set[str] | None = None, force: bool = False,
                 placeholders_only: bool = False) -> None:
    timeline = load_timeline()
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for item in timeline:
        sid = item["sid"]
        if only and sid not in only:
            continue
        out = SHOTS_DIR / f"{sid}.mp4"
        mod = VIDEO_DIR / "shots" / f"{item['module']}.py"
        if placeholders_only or not mod.exists():
            print(f"[{sid}] MISSING renderer {mod.name} -- placeholder")
            placeholder(sid, item, out)
            continue
        if out.exists() and not force and out.stat().st_mtime > mod.stat().st_mtime:
            if abs(probe_duration(out) - item["duration"]) < 0.05:
                print(f"[{sid}] up to date")
                continue
        py = VIDEO_PY if item["interp"] == "video" else REPO_PY
        cmd = [str(py), str(mod), "--out", str(out), "--duration", str(item["duration"])]
        cmd += [str(a) for a in item["args"]]
        print(f"[{sid}] rendering ({item['duration']:.1f}s) -> {out.name}")
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            # One broken renderer must not abort the other sixteen; stub it and keep going so the
            # cut stays watchable, and list the failures at the end.
            print(f"[{sid}] RENDER FAILED (exit {r.returncode}) -- placeholder")
            placeholder(sid, item, out)
            failed.append(sid)
            continue
        d = probe_duration(out)
        if abs(d - item["duration"]) > 0.08:
            print(f"[{sid}] WARNING duration {d:.2f}s != {item['duration']:.2f}s")
    if failed:
        print(f"\nFAILED SHOTS: {', '.join(failed)}")


def placeholder(sid: str, item: dict, out: Path) -> None:
    import style
    txt = f"{sid} :: {item['module']}".replace(":", r"\:")
    subprocess.run([
        str(FFMPEG), "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c={style.BG}:s={style.WIDTH}x{style.HEIGHT}:r={style.FPS}",
        "-t", str(item["duration"]),
        "-vf", f"drawtext=text='{txt}':fontcolor={style.FG_MUTED}:fontsize=48:"
               f"x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", str(out),
    ], check=True)


def probe_duration(path: Path) -> float:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return -1.0


# --------------------------------------------------------------------------- assembly

def build_audio(timeline: list[dict]) -> Path:
    """Narration laid on a silent bed at each shot's start time. No music, no ambience."""
    total = sum(i["duration"] for i in timeline)
    inputs, filt, mixers = [], [], []
    inputs += ["-f", "lavfi", "-t", f"{total}", "-i", "anullsrc=r=48000:cl=stereo"]
    mixers.append("[0:a]")
    idx = 1
    for item in timeline:
        wav = AUDIO_DIR / f"{item['sid']}.wav"
        if not wav.exists():
            continue
        delay = int(round((item["start"] + item["audio_start"]) * 1000))
        inputs += ["-i", str(wav)]
        filt.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
                    f"adelay={delay}|{delay},volume=1.0[a{idx}]")
        mixers.append(f"[a{idx}]")
        idx += 1
    # The author asked for the deployment capture's audio to be dropped, so the mix is
    # narration only -- no ambience bed under s16.
    filt.append("".join(mixers) + f"amix=inputs={len(mixers)}:normalize=0:dropout_transition=0,"
                                  f"loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=limit=0.95,atrim=0:{total},asetpts=N/SR/TB[aout]")
    out = BUILD / "narration_mix.m4a"
    subprocess.run([str(FFMPEG), "-y", "-v", "error", *inputs,
                    "-filter_complex", ";".join(filt), "-map", "[aout]",
                    "-c:a", "aac", "-b:a", "160k", str(out)], check=True)
    return out


def concat_video(timeline: list[dict]) -> Path:
    lst = BUILD / "concat.txt"
    lst.write_text("".join(f"file '{(SHOTS_DIR / (i['sid'] + '.mp4')).as_posix()}'\n"
                           for i in timeline))
    out = BUILD / "silent.mp4"
    # Re-encode rather than stream-copy: the 17 shots come from different renderers (manim,
    # FrameWriter, ffmpeg filter graphs) and a stream copy would inherit whichever SPS/PPS
    # mismatch they happen to have. This intermediate is near-lossless; the deliverable is
    # re-encoded from it anyway.
    subprocess.run([str(FFMPEG), "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-fps_mode", "cfr", "-r", str(style_fps()),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def style_fps() -> int:
    import style
    return style.FPS


def encode(video: Path, audio: Path, out: Path, subs: Path | None,
           target_bytes: int = MAX_BYTES) -> Path:
    """Two-pass H.264 sized to fit the 20 MB limit with ~7 % headroom."""
    import style
    dur = probe_duration(video)
    audio_kbps = 160
    total_kbps = (target_bytes * 0.92 * 8 / 1000) / dur
    v_kbps = int(max(total_kbps - audio_kbps, 500))
    vf = []
    if subs is not None:
        vf.append(f"ass={subs.as_posix()}")
    vf.append("format=yuv420p")
    passlog = str(BUILD / "x264")
    common = [str(FFMPEG), "-y", "-v", "error", "-i", str(video), "-i", str(audio),
              "-map", "0:v:0", "-map", "1:a:0", "-vf", ",".join(vf),
              "-c:v", "libx264", "-preset", "slower", "-b:v", f"{v_kbps}k",
              "-maxrate", f"{int(v_kbps * 1.5)}k", "-bufsize", f"{int(v_kbps * 3)}k",
              "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
              "-x264-params", "interlaced=0", "-passlogfile", passlog]
    subprocess.run(common + ["-pass", "1", "-an", "-f", "mp4", "/dev/null"], check=True)
    subprocess.run(common + ["-pass", "2", "-c:a", "aac", "-b:a", f"{audio_kbps}k",
                             "-movflags", "+faststart", str(out)], check=True)
    for p in BUILD.glob("x264*"):
        p.unlink()
    return out


def assemble() -> None:
    timeline = load_timeline()
    OUT.mkdir(parents=True, exist_ok=True)
    srt = OUT / "narration.srt"
    tts.build_srt(timeline, srt)
    tts.build_vtt(srt, OUT / "narration.vtt")
    ass = BUILD / "narration.ass"
    tts.build_ass(timeline, ass, font=__import__("style").FONT)
    (OUT / "narration_script.md").write_text(script_markdown(timeline), encoding="utf-8")

    silent = concat_video(timeline)
    audio = build_audio(timeline)
    encode(silent, audio, OUT / "icra_video_subtitled.mp4", ass)
    encode(silent, audio, OUT / "icra_video.mp4", None)
    check()


def script_markdown(timeline: list[dict]) -> str:
    rows = ["# Narration script\n",
            "| # | t | shot | narration |", "|---|---|------|-----------|"]
    for i in timeline:
        m, s = divmod(i["start"], 60)
        rows.append(f"| {i['sid']} | {int(m)}:{s:04.1f} | {i['module']} | {i['text']} |")
    return "\n".join(rows) + "\n"


def check() -> None:
    print("\nICRA constraint check")
    ok = True
    for f in sorted(OUT.glob("*.mp4")):
        info = json.loads(subprocess.run(
            [str(FFPROBE), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(f)],
            capture_output=True, text=True).stdout)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        dur, size = float(info["format"]["duration"]), int(info["format"]["size"])
        num, den = (int(x) for x in v["r_frame_rate"].split("/"))
        fps = num / den
        prog = v.get("field_order", "progressive") in ("progressive", "unknown")
        good = (dur <= MAX_SECONDS and size <= MAX_BYTES and int(v["height"]) >= MIN_HEIGHT
                and fps >= MIN_FPS and prog and v["codec_name"] == "h264")
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {f.name}: {dur:.1f}s, {size/1e6:.1f} MB, "
              f"{v['width']}x{v['height']}, {fps:.0f} fps, {v['codec_name']}, "
              f"{'progressive' if prog else v.get('field_order')}")
    print("  all deliverables conform" if ok else "  CONSTRAINT VIOLATION")


def fetch_media() -> None:
    """Copy every external asset the film uses into video/media (self-contained + reproducible)."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    srcs = {
        "realworld_source.mp4": REPO / "video_material/ssm_lab_recording_cut_with_audio.mp4",
    }
    for dst, src in srcs.items():
        if src.exists() and not (MEDIA / dst).exists():
            shutil.copy2(src, MEDIA / dst)
            print(f"copied {src} -> media/{dst}")
    tts.fetch_voice()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "fetch", "narrate", "shots", "assemble", "check"])
    ap.add_argument("--only", default="", help="comma-separated shot ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--placeholders", action="store_true",
                    help="stub every shot (integration test of the assembly path)")
    a = ap.parse_args()
    only = {s.strip() for s in a.only.split(",") if s.strip()} or None
    if a.stage in ("all", "fetch"):
        fetch_media()
    if a.stage in ("all", "narrate"):
        narrate(force=a.force)
    if a.stage in ("all", "shots"):
        render_shots(only, force=a.force, placeholders_only=a.placeholders)
    if a.stage in ("all", "assemble"):
        assemble()
    if a.stage == "check":
        check()


if __name__ == "__main__":
    main()

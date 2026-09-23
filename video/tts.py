"""Offline narration synthesis (Piper) + SRT generation.

Voice: en_US-ryan-high, downloaded once into video/assets/voices by `build.py fetch`.
Everything is local, so the narration is reproducible bit-for-bit.
"""
from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent
VOICE = VIDEO_DIR / "assets/voices/en_US-ryan-high.onnx"
VOICE_URL = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/"
             "en_US-ryan-high.onnx")

# Piper says "YOLO" letter-by-letter and mangles a few technical tokens; spell them phonetically
# in the *spoken* text only -- the subtitles keep the written form.
SPOKEN_FIXES = [
    (r"\bYOLO\b", "Yolo"),
    (r"\bSARA\b", "Sarah"),
    (r"\bISO\b", "I S O"),
    (r"\bGPU\b", "G P U"),
    (r"\bhertz\b", "hurts"),
    (r"\bPL\b", "P L"),
]


def fetch_voice() -> Path:
    VOICE.parent.mkdir(parents=True, exist_ok=True)
    if not VOICE.exists():
        subprocess.run(["curl", "-sL", "-o", str(VOICE), VOICE_URL], check=True)
        subprocess.run(["curl", "-sL", "-o", str(VOICE) + ".json", VOICE_URL + ".json"], check=True)
    return VOICE


def _spoken(text: str) -> str:
    for pat, rep in SPOKEN_FIXES:
        text = re.sub(pat, rep, text)
    return text


def synthesize(text: str, out_wav: Path, length_scale: float = 1.0) -> float:
    """Write `text` to `out_wav`; return its duration in seconds."""
    from piper import PiperVoice, SynthesisConfig

    global _VOICE_CACHE
    try:
        voice = _VOICE_CACHE
    except NameError:
        voice = _VOICE_CACHE = PiperVoice.load(str(fetch_voice()))
    cfg = SynthesisConfig(length_scale=length_scale, noise_scale=0.6, noise_w_scale=0.75)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        voice.synthesize_wav(_spoken(text), w, syn_config=cfg)
    return wav_duration(out_wav)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# --------------------------------------------------------------------------- subtitles

def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_caption(text: str, max_chars: int = 84, max_lines: int = 2) -> list[str]:
    """Split narration into caption chunks of <= max_lines lines of <= max_chars/max_lines."""
    words, chunks, cur = text.split(), [], ""
    limit = max_chars
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) > limit and cur:
            chunks.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


def wrap_two_lines(chunk: str) -> str:
    """Balance a chunk over at most two lines."""
    words = chunk.split()
    if len(" ".join(words)) <= 44:
        return " ".join(words)
    best, best_cost = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        cost = abs(len(a) - len(b)) + 100 * max(0, max(len(a), len(b)) - 52)
        if best_cost is None or cost < best_cost:
            best, best_cost = (a, b), cost
    return "\n".join(best)


def build_srt(timeline: list[dict], out_srt: Path) -> None:
    """timeline: [{start, duration, audio_duration, text}] -> SRT with per-chunk timing.

    Chunks of one shot share the shot's spoken span, weighted by character count, so the caption
    tracks the voice closely without needing forced alignment.
    """
    lines, idx = [], 1
    for item in timeline:
        text = item["text"].strip()
        if not text:
            continue
        span = max(item.get("audio_duration") or item["duration"], 0.1)
        chunks = split_caption(text)
        total = sum(len(c) for c in chunks)
        t = item["start"]
        for c in chunks:
            d = span * len(c) / total
            lines.append(f"{idx}\n{_srt_time(t)} --> {_srt_time(t + d)}\n{wrap_two_lines(c)}\n")
            idx += 1
            t += d
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    out_srt.write_text("\n".join(lines), encoding="utf-8")


def build_vtt(srt: Path, out_vtt: Path) -> None:
    body = srt.read_text(encoding="utf-8").replace(",", ".")
    out_vtt.write_text("WEBVTT\n\n" + body, encoding="utf-8")


def dump_timeline(timeline: list[dict], out_json: Path) -> None:
    out_json.write_text(json.dumps(timeline, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- ASS subtitles

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{font},{size},&H00ECEFF4,&H00ECEFF4,&H64000000,&H00000000,0,0,0,0,100,100,0.6,0,1,3.2,0,2,160,160,{marginv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text
"""


def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass(timeline: list[dict], out_ass, font: str = "Inter", size: int = 40,
              marginv: int = 64) -> None:
    """Burn-in subtitle track at the film's real resolution (libass defaults to 288p otherwise)."""
    from pathlib import Path
    out_ass = Path(out_ass)
    lines = [ASS_HEADER.format(font=font, size=size, marginv=marginv)]
    for item in timeline:
        text = item["text"].strip()
        if not text:
            continue
        span = max(item.get("audio_duration") or item["duration"], 0.1)
        chunks = split_caption(text, max_chars=96)
        total = sum(len(c) for c in chunks)
        t = item["start"]
        for c in chunks:
            d = span * len(c) / total
            body = wrap_two_lines(c).replace("\n", r"\N")
            lines.append(f"Dialogue: 0,{_ass_time(t)},{_ass_time(t + d)},Sub,,0,0,,{body}")
            t += d
    out_ass.write_text("\n".join(lines) + "\n", encoding="utf-8")

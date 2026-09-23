"""Re-time the narration script against a *dubbed* cut, and emit YouTube subtitles.

The film's own subtitles are derived from the synthesised narration, so once the voice-over is
re-recorded by hand the timings in `out/narration_script.md` no longer match the audio. This
script recovers them from the audio itself:

  1. transcribe the dubbed audio with word-level timestamps (faster-whisper),
  2. force-align the *known* script text to that transcript (difflib on normalised words), so
     wording comes from the script and timing comes from the audio -- ASR mistakes cannot change
     a single word of what the subtitles say,
  3. write the corrected script table and sentence-level SRT / VTT / plain-text transcript.

    python video/retime_subtitles.py --video out/ICRA_2027.mp4

Outputs (next to the video, in `video/out/`):
    narration_script.md      the same table, with timings measured from the audio
    ICRA_2027.srt            YouTube-ready subtitles (also accepted by every player)
    ICRA_2027.vtt            WebVTT
    ICRA_2027.transcript.txt plain transcript, for YouTube's "auto-sync" upload path
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VIDEO_DIR))

FFMPEG = "/home/thumm/miniconda3/envs/chmp-video/bin/ffmpeg"
MAX_CHARS = 84          # per caption chunk (two balanced lines)
MAX_LINE = 46           # per line
MIN_SEC_PER_WORD = 0.12  # below this a 'gap' cannot really hold the words (>8 w/s)


# --------------------------------------------------------------------------- the written script

def parse_script(md: Path) -> list[dict]:
    """Rows of `out/narration_script.md` -> [{sid, module, text}] (the wording is authoritative)."""
    rows = []
    for line in md.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| s"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "#":
            continue
        if len(cells) >= 5:      # already re-timed: | sid | start | end | shot | narration |
            rows.append({"sid": cells[0], "module": cells[3], "text": cells[4]})
        elif len(cells) == 4:    # as written by build.py: | sid | t | shot | narration |
            rows.append({"sid": cells[0], "module": cells[2], "text": cells[3]})
    return rows


# --------------------------------------------------------------------------- the spoken audio

def transcribe(video: Path, model_size: str, refresh: bool = False) -> list[dict]:
    """[{word, start, end}] over the whole film, cached beside the video."""
    from faster_whisper import WhisperModel

    cache = video.with_suffix(f".words.{model_size}.json")
    if cache.exists() and not refresh:
        print(f"using cached transcript {cache.name} (--refresh to redo)")
        return json.loads(cache.read_text())

    wav = Path("/tmp/_retime_audio.wav")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # vad_filter=False on purpose: the VAD was observed to swallow a whole sentence of real
    # speech, which then looks exactly like "the script says something the dub does not".
    segments, _ = model.transcribe(str(wav), language="en", word_timestamps=True,
                                   vad_filter=False, beam_size=5)
    out = []
    for seg in segments:
        for w in seg.words or []:
            out.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    cache.write_text(json.dumps(out))
    return out


# --------------------------------------------------------------------------- alignment

def norm(w: str) -> str:
    """Compare on letters/digits only, and spell digits out so '400' matches 'four hundred'."""
    w = re.sub(r"[^a-z0-9]", "", w.lower())
    return w


NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}


def align(script_words: list[str], asr: list[dict]) -> tuple[list[tuple[float, float]], int]:
    """A (start, end) for every script word, or None where the ASR had nothing to match.

    difflib over normalised tokens; unmatched runs are linearly interpolated between their
    matched neighbours, so a word the recogniser dropped still lands in the right place.
    """
    import difflib

    a = [norm(w) for w in script_words]
    b = [norm(w["word"]) for w in asr]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    times: list[tuple[float, float] | None] = [None] * len(a)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            times[i + k] = (asr[j + k]["start"], asr[j + k]["end"])

    n_matched = sum(1 for t in times if t is not None)

    # A run of unmatched script words bounded by two anchors has only so much time available.
    # If fitting it there would need an impossible speaking rate, the words are not in the audio
    # (the script and the dub disagree) -- flag them rather than crush them into the gap.
    unspoken: list[int] = []
    anchors = [i for i, t in enumerate(times) if t is not None]
    i = 0
    while i < len(times):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        prev = max((k for k in anchors if k < i), default=None)
        nxt = min((k for k in anchors if k >= j), default=None)
        if prev is not None and nxt is not None:
            available = times[nxt][0] - times[prev][1]
            if available < (j - i) * MIN_SEC_PER_WORD:
                unspoken.extend(range(i, j))
        i = j

    # interpolate the gaps
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        raise SystemExit("alignment failed: no script word matched the transcript")
    for i in range(len(times)):
        if times[i] is not None:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        if prev is None:
            times[i] = (times[nxt][0], times[nxt][0])
        elif nxt is None:
            times[i] = (times[prev][1], times[prev][1])
        else:
            t0, t1 = times[prev][1], times[nxt][0]
            f0 = (i - prev) / (nxt - prev)
            f1 = (i + 1 - prev) / (nxt - prev)
            times[i] = (t0 + (t1 - t0) * f0, t0 + (t1 - t0) * f1)
    return times, n_matched, unspoken


# --------------------------------------------------------------------------- caption chunking

def sentences(text: str) -> list[str]:
    """Split on sentence enders, keeping the punctuation, then on ';' / ':' if still too long."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for p in parts:
        if len(p) <= MAX_CHARS * 2:
            out.append(p)
            continue
        sub = re.split(r"(?<=[;:,])\s+", p)
        cur = ""
        for s in sub:
            if cur and len(cur) + len(s) + 1 > MAX_CHARS * 2:
                out.append(cur)
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            out.append(cur)
    return [s for s in out if s]


def chunk(sentence: str) -> list[str]:
    """A sentence -> caption-sized pieces, balanced.

    Filling greedily to MAX_CHARS leaves stubs ("instance." on its own for 0.6 s), so the number
    of pieces is decided first and the words are then spread evenly over them.
    """
    words = sentence.split()
    if not words:
        return []
    n = max(1, -(-len(sentence) // MAX_CHARS))          # ceil
    target = len(sentence) / n
    out, cur = [], ""
    for i, w in enumerate(words):
        cand = (cur + " " + w).strip()
        remaining = len(words) - i - 1
        # close the piece once it has reached its share, but never strand the tail
        # a comma/colon just before the ideal split is a better place to break than the
        # middle of a phrase, so close a little early when one is within reach
        early = (cur.endswith((",", ";", ":")) and len(cand) > target * 0.72)
        if cur and (len(cand) > target or early) and len(out) < n - 1 and remaining >= 1:
            out.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def wrap(chunk_text: str) -> str:
    """Balance a chunk over at most two lines."""
    words = chunk_text.split()
    if len(chunk_text) <= MAX_LINE:
        return chunk_text
    best, cost_best = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        cost = abs(len(a) - len(b)) + 200 * max(0, max(len(a), len(b)) - MAX_LINE)
        if a.endswith((",", ";", ":")):          # prefer breaking at a phrase boundary
            cost -= 14
        if b.split()[0].lower() in ("the", "a", "an", "of", "to", "in", "and", "with", "that"):
            cost += 6                            # avoid orphaning a function word
        if cost_best is None or cost < cost_best:
            best, cost_best = (a, b), cost
    return "\n".join(best)


# --------------------------------------------------------------------------- output

def ts_srt(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_vtt(t: float) -> str:
    return ts_srt(t).replace(",", ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=VIDEO_DIR / "out/ICRA_2027.mp4")
    ap.add_argument("--script", type=Path, default=VIDEO_DIR / "out/narration_script.md")
    ap.add_argument("--model", default="medium.en", help="faster-whisper model size")
    ap.add_argument("--min-dur", type=float, default=1.2, help="minimum caption duration (s)")
    ap.add_argument("--refresh", action="store_true", help="re-transcribe")
    ap.add_argument("--lead", type=float, default=0.10, help="show captions this early (s)")
    a = ap.parse_args()

    rows = parse_script(a.script)
    asr = transcribe(a.video, a.model, refresh=a.refresh)
    print(f"{len(rows)} script rows, {len(asr)} recognised words")

    # one flat word list, remembering which row each word came from
    flat, owner = [], []
    for ri, r in enumerate(rows):
        for w in r["text"].split():
            flat.append(w)
            owner.append(ri)
    times, n_matched, unspoken = align(flat, asr)
    unspoken_set = set(unspoken)
    if unspoken:
        runs, cur = [], [unspoken[0]]
        for k in unspoken[1:]:
            (cur.append(k) if k == cur[-1] + 1 else (runs.append(cur), cur.clear(), cur.append(k)))
        runs.append(cur)
        print("\nWARNING -- script text that is NOT in the dubbed audio "
              "(omitted from the subtitles):")
        for r in runs:
            print(f"  {rows[owner[r[0]]]['sid']}: \"{' '.join(flat[k] for k in r)}\"")
        print()
    print(f"aligned {n_matched}/{len(flat)} script words directly "
          f"({100 * n_matched / len(flat):.1f} %); the rest interpolated between neighbours")

    # ---- per-row spans, for the corrected script table
    for ri, r in enumerate(rows):
        idx = [i for i, o in enumerate(owner) if o == ri and i not in unspoken_set]
        r["unspoken"] = " ".join(flat[i] for i, o in enumerate(owner)
                                 if o == ri and i in unspoken_set)
        if idx and r["text"]:
            r["start"] = times[idx[0]][0]
            r["end"] = times[idx[-1]][1]
        else:
            r["start"] = r["end"] = None

    # ---- captions
    caps = []
    for ri, r in enumerate(rows):
        if not r["text"]:
            continue
        idx = [i for i, o in enumerate(owner) if o == ri]
        pos = 0
        for sent in sentences(r["text"]):
            for piece in chunk(sent):
                n = len(piece.split())
                span = idx[pos:pos + n]
                pos += n
                spoken = [k for k in span if k not in unspoken_set]
                if not spoken:
                    continue
                words = [flat[k] for k in spoken]
                caps.append({"text": wrap(" ".join(words)),
                             "start": times[spoken[0]][0],
                             "end": times[spoken[-1]][1]})

    # tidy, in this order of priority: never overlap > reach min duration > lead-in
    for c in caps:
        c["start"] = max(0.0, c["start"] - a.lead)
    # Absorb captions that cannot be on screen long enough to read.  What limits a caption is
    # not its own word span but the window until the next one starts, so that is what is tested.
    def window(i: int, cs: list[dict]) -> float:
        return (cs[i + 1]["start"] if i + 1 < len(cs) else cs[i]["end"]) - cs[i]["start"]

    changed = True
    while changed and len(caps) > 1:
        changed = False
        for i in range(1, len(caps)):
            flat_prev = caps[i - 1]["text"].replace("\n", " ")
            flat_cur = caps[i]["text"].replace("\n", " ")
            if (window(i, caps) < a.min_dur
                    and len(flat_prev) + len(flat_cur) + 1 <= MAX_CHARS):
                caps[i - 1]["text"] = wrap(flat_prev + " " + flat_cur)
                caps[i - 1]["end"] = caps[i]["end"]
                caps.pop(i)
                changed = True
                break

    # Ends, in priority order: never overlap the next caption > reach min duration.
    for i, c in enumerate(caps):
        c["end"] = max(c["end"], c["start"] + a.min_dur)
        if i + 1 < len(caps):                       # one frame of gap, never an overlap
            c["end"] = min(c["end"], caps[i + 1]["start"] - 0.04)

    out = a.video.parent
    stem = a.video.stem
    (out / f"{stem}.srt").write_text(
        "\n".join(f"{i+1}\n{ts_srt(c['start'])} --> {ts_srt(c['end'])}\n{c['text']}\n"
                  for i, c in enumerate(caps)) + "\n", encoding="utf-8")
    (out / f"{stem}.vtt").write_text(
        "WEBVTT\n\n" + "\n".join(f"{ts_vtt(c['start'])} --> {ts_vtt(c['end'])}\n{c['text']}\n"
                                 for c in caps) + "\n", encoding="utf-8")
    (out / f"{stem}.transcript.txt").write_text(
        "\n".join(r["text"] for r in rows if r["text"]) + "\n", encoding="utf-8")

    # ---- corrected script table
    head = ["# Narration script",
            "",
            f"Timings measured from `{a.video.name}` (audio-aligned, "
            f"faster-whisper `{a.model}` + forced alignment to this text).",
            "",
            "| # | start | end | shot | narration |",
            "|---|-------|-----|------|-----------|"]
    body = []
    for r in rows:
        if r["start"] is None:
            body.append(f"| {r['sid']} | — | — | {r['module']} | {r['text']} |")
        else:
            m0, s0 = divmod(r["start"], 60)
            m1, s1 = divmod(r["end"], 60)
            note = (f" **[not in the audio: “{r['unspoken']}”]**" if r["unspoken"] else "")
            body.append(f"| {r['sid']} | {int(m0)}:{s0:05.2f} | {int(m1)}:{s1:05.2f} | "
                        f"{r['module']} | {r['text']}{note} |")
    a.script.write_text("\n".join(head + body) + "\n", encoding="utf-8")

    (out / f"{stem}.alignment.json").write_text(json.dumps(
        {"video": a.video.name, "model": a.model, "captions": caps,
         "rows": [{k: r[k] for k in ("sid", "module", "start", "end")} for r in rows]},
        indent=2), encoding="utf-8")

    print(f"wrote {out/f'{stem}.srt'}\n      {out/f'{stem}.vtt'}\n"
          f"      {out/f'{stem}.transcript.txt'}\n      {a.script}")
    print(f"{len(caps)} captions, last ends at {caps[-1]['end']:.2f} s")


if __name__ == "__main__":
    main()

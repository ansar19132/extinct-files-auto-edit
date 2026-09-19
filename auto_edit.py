#!/usr/bin/env python3
"""
The Extinct Files — Auto Edit Tool
Input:  script.txt (one narration sentence per line)
        voiceover.mp3
        images/ (one image per script line, sorted)
        music.mp3 (optional background music)
Output: output/long_final.mp4  (16:9, 1080p, captions, name cards, ducked music)
        output/short_1..3.mp4   (9:16 vertical Shorts, 30-40s each)

Usage:
    .venv/bin/python auto_edit.py --project /path/to/project [--model small] [--no-shorts]
"""
import argparse, difflib, json, os, re, subprocess, sys
from pathlib import Path

FPS = 30
W, H = 1920, 1080
SHORT_W, SHORT_H = 1080, 1920

# font locations (overridable, e.g. by the web app)
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTS_DIR = "/usr/share/fonts"

# ---------------------------------------------------------------- config
DEFAULT_CONFIG = {
    "xfade_duration": 0.5,          # normal transition length (seconds)
    "section_xfade_duration": 0.9,  # dip-to-black between creatures
    "section_pattern": r"^number\s+\w+\s*:",
    "music_volume": 0.15,
    "kenburns_cycle": ["zin", "zout", "panr", "panl"],
    "caption_fontsize": 64,
    "shorts": [                     # 1-based [from_sentence, to_sentence]
        [1, 4],
        [16, 19],
        [37, 40],
    ],
    "short_max_seconds": 40,
}

# ---------------------------------------------------------------- helpers
def norm_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def ts(seconds):
    """seconds -> ASS timestamp h:mm:ss.cc"""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600); m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def run(cmd, **kw):
    print("+", " ".join(cmd[:4]), "..." if len(cmd) > 4 else "")
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit(f"command failed: {cmd[0]}")
    return r

# ---------------------------------------------------------------- alignment
def resolve_model(model_name):
    """Use a pre-downloaded local model if present (no download needed)."""
    local = Path(f"/opt/hatch-image/models/asr/faster-whisper-{model_name}")
    if local.is_dir() and (local / "model.bin").exists():
        print(f"      using local model: {local}")
        return str(local)
    return model_name

def transcribe_words(voiceover, model_name="small"):
    from faster_whisper import WhisperModel
    print(f"[1/5] Transcribing voiceover ({model_name})...")
    model = WhisperModel(resolve_model(model_name), device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(voiceover), word_timestamps=True, language="en")
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"w": norm_text(w.word), "start": w.start, "end": w.end})
    words = [x for x in words if x["w"]]
    print(f"      {len(words)} words transcribed")
    if not words:
        raise SystemExit("No words transcribed — check the voiceover file.")
    return words

def align_sentences(script_lines, twords):
    """Map each script sentence -> (start, end) in audio via word alignment."""
    import difflib
    print("[2/5] Aligning sentences to voiceover...")
    # flat script word list with sentence boundaries
    sw, bounds = [], []
    for i, line in enumerate(script_lines):
        ws = norm_text(line).split()
        bounds.append((len(sw), len(sw) + len(ws), i))
        sw.extend(ws)
    tw = [x["w"] for x in twords]
    sm = difflib.SequenceMatcher(None, sw, tw, autojunk=False)
    # map script word idx -> transcript word idx via equal blocks (interpolate gaps)
    mapping = {}
    prev_si, prev_ti = 0, 0
    for tag, a, b, c, d in sm.get_opcodes():
        if tag == "equal":
            # interpolate gap between prev block end and this block start
            gap_s = a - prev_si
            gap_t = c - prev_ti
            for k in range(gap_s):
                frac = (k + 1) / (gap_s + 1)
                mapping[prev_si + k] = prev_ti + int(round(frac * gap_t))
            for k in range(b - a):
                mapping[a + k] = c + k
            prev_si, prev_ti = b, d
    # tail gap
    gap_s = len(sw) - prev_si
    gap_t = len(tw) - prev_ti
    for k in range(gap_s):
        frac = (k + 1) / (gap_s + 1)
        mapping[prev_si + k] = min(prev_ti + int(round(frac * gap_t)), len(tw) - 1)

    sentences = []
    for s0, s1, i in bounds:
        t0 = max(0, mapping.get(s0, 0))
        t1 = min(len(twords) - 1, mapping.get(s1 - 1, len(twords) - 1))
        if t1 < t0:
            t0, t1 = t0, t0
        start = max(0.0, twords[t0]["start"] - 0.12)
        end = twords[t1]["end"] + 0.30
        if end - start < 1.0:
            end = start + 1.0
        sentences.append({"line": script_lines[i], "start": start, "end": end,
                          "w0": t0, "w1": t1})
    # enforce monotonic order
    for i in range(1, len(sentences)):
        if sentences[i]["start"] < sentences[i - 1]["end"] - 0.05:
            sentences[i]["start"] = sentences[i - 1]["end"] - 0.05
        if sentences[i]["end"] <= sentences[i]["start"]:
            sentences[i]["end"] = sentences[i]["start"] + 1.0
    # per-sentence display durations
    for s in sentences:
        s["dur"] = s["end"] - s["start"]
    total_audio = twords[-1]["end"]
    print(f"      {len(sentences)} sentences aligned, audio {total_audio:.1f}s")
    return sentences, twords

# ---------------------------------------------------------------- captions (ASS karaoke)
def build_caption_words(sentences, twords):
    """Per-sentence caption words: script spelling where close to what was
    heard (fixes Whisper misspellings like 'helicopion'), transcript timing."""
    for s in sentences:
        sw_orig = s["line"].split()
        # norm_text can split one script word into several normalized tokens
        # ("soft-bodied" -> "soft bodied") or drop tokens entirely ("-"),
        # so keep a map from each normalized token back to its script word.
        sw_norm = []
        norm_to_orig = []
        for oi, w in enumerate(sw_orig):
            for p in norm_text(w).split():
                sw_norm.append(p)
                norm_to_orig.append(oi)
        tw = twords[s["w0"]:s["w1"] + 1]
        sm = difflib.SequenceMatcher(None, sw_norm, [t["w"] for t in tw], autojunk=False)
        caps = []
        for tag, a, b, c, d in sm.get_opcodes():
            if tag == "equal":
                for k in range(b - a):
                    oi = norm_to_orig[a + k]
                    caps.append((sw_orig[oi], tw[c + k]["start"], tw[c + k]["end"]))
            elif tag == "replace" and (b - a) == (d - c):
                for k in range(b - a):
                    oi = norm_to_orig[a + k]
                    r = difflib.SequenceMatcher(None, sw_norm[a + k], tw[c + k]["w"]).ratio()
                    txt = sw_orig[oi] if r > 0.55 else tw[c + k]["w"]
                    caps.append((txt, tw[c + k]["start"], tw[c + k]["end"]))
            elif tag == "insert":
                for k in range(d - c):
                    caps.append((tw[c + k]["w"], tw[c + k]["start"], tw[c + k]["end"]))
            # "delete": script word not spoken -> skip it
        if not caps:
            caps = [(w, t["start"], t["end"]) for w, t in zip(sw_orig, tw)]
        # merge repeats from one script word split into several norm tokens
        merged = []
        for txt, st, en in caps:
            if merged and merged[-1][0] == txt:
                merged[-1] = (txt, merged[-1][1], en)
            else:
                merged.append((txt, st, en))
        s["caps"] = merged

def write_ass(path, sentences, play_w, play_h, fontsize, margin_v):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,{fontsize},&H00FFFFFF,&H00003CFF,&H90000000,&H90000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # SecondaryColour &H00003CFF -> karaoke fill orange-yellow (AABBGGRR)
    events = []
    for s in sentences:
        parts = []
        for (txt, st, en) in s["caps"]:
            cs = max(1, int(round((en - st) * 100)))
            parts.append("{\\k%d}%s " % (cs, txt))
        text = "".join(parts).strip()
        est, een = s["caps"][0][1], s["caps"][-1][2]
        events.append(
            f"Dialogue: 0,{ts(est)},{ts(een)},Cap,,0,0,0,,{text}"
        )
    path.write_text(head + "\n".join(events) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- video build
def kenburns(i, dur, style, F, ow=W, oh=H):
    sw, sh = int(ow * 1.34), int(oh * 1.34)
    base = f"scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},setsar=1"
    if style == "zin":
        zb = f"z='1+0.14*on/{F}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    elif style == "zout":
        zb = f"z='1.14-0.14*on/{F}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    elif style == "panr":
        zb = f"z=1.14:x='(iw-iw/zoom)*on/{F}':y='ih/2-(ih/zoom/2)'"
    else:  # panl
        zb = f"z=1.14:x='(iw-iw/zoom)*(1-on/{F})':y='ih/2-(ih/zoom/2)'"
    return f"[{i}:v]{base},zoompan={zb}:d={F}:s={ow}x{oh}:fps={FPS},setsar=1[v{i}]"

def parse_name_card(line, pattern):
    m = re.match(pattern, line, flags=re.I)
    if not m:
        return None
    rest = line[m.end():].strip()
    name = rest.split(",")[0].strip().upper()
    name = re.sub(r"\s+", " ", name)
    return name or None

def build_filter(sentences, cfg, total_dur, has_music, D, dx, is_section, audio_t0,
                 ow=W, oh=H, ass_path=None):
    # D[i]      = display duration of image i (start-to-start; last incl. tail)
    # dx[i]     = transition duration after clip i (into sentence i+1)
    # Video timeline == audio timeline: transition i sits at the start of
    # sentence i+1, so every image is fully visible during its own narration.
    filt = []
    cycle = cfg["kenburns_cycle"]
    n = len(sentences)
    # per-image ken burns (clips extended by the following transition length)
    for i in range(n):
        ext = dx[i] if i < n - 1 else 0.0
        F = max(2, int(round((D[i] + ext) * FPS)))
        filt.append(kenburns(i, D[i], cycle[i % len(cycle)], F, ow, oh))
    # xfade chain
    prev = "v0"
    cum = D[0]
    for i in range(n - 1):
        trans = "fadeblack" if is_section[i + 1] else "fade"
        filt.append(f"[{prev}][v{i + 1}]xfade=transition={trans}:duration={dx[i]}:offset={cum:.3f}[x{i + 1}]")
        prev = f"x{i + 1}"
        if i < n - 2:
            cum += D[i + 1]
    vout = prev
    # global timeline per sentence (for name cards)
    gstarts = [0.0]
    for i in range(n - 1):
        gstarts.append(gstarts[-1] + D[i])
    # name cards via drawtext on section starts
    card_y = oh - (300 if oh <= 1080 else 560)
    for i, s in enumerate(sentences):
        name = parse_name_card(s["line"], cfg["section_pattern"])
        if name:
            st = gstarts[i] + 0.4
            en = st + 2.6
            safe = name.replace("'", "").replace(":", "")
            filt.append(
                f"[{vout}]drawtext=fontfile={FONT_BOLD}"
                f":text='{safe}':fontsize=92:fontcolor=white:borderw=3:bordercolor=black"
                f":x=(w-text_w)/2:y={card_y}"
                f":enable='between(t,{st:.2f},{en:.2f})'"
                f":alpha='if(lt(t,{st:.2f}+0.4),(t-{st:.2f})/0.4,if(gt(t,{en:.2f}-0.4),({en:.2f}-t)/0.4,1))'"
                f"[vout{i}]"
            )
            vout = f"vout{i}"
    filt.append(f"[{vout}]format=yuv420p[vfinal]")
    if ass_path:
        filt.append(f"[vfinal]subtitles='{ass_path}':fontsdir='{FONTS_DIR}'[vout]")
    else:
        filt.append("[vfinal]null[vout]")
    # audio (trim leading silence so audio time == video time)
    n = len(sentences)
    a_in = n       # voiceover input index
    filt.append(
        f"[{a_in}:a]atrim=start={audio_t0:.2f},asetpts=PTS-STARTPTS,aresample=48000,"
        f"apad,atrim=0:{total_dur:.2f}[voice]"
    )
    if has_music:
        m_in = n + 1
        filt.append(
            f"[{m_in}:a]aresample=48000,atrim=0:{total_dur:.2f},asetpts=PTS-STARTPTS,"
            f"volume={cfg['music_volume']}[mm]"
        )
        filt.append(
            "[mm][voice]sidechaincompress=threshold=0.03:ratio=10:attack=300:release=1200[ducked]"
        )
        filt.append(
            "[voice][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
            "alimiter=limit=0.95,loudnorm=I=-16:TP=-1.5:LRA=11[afinal]"
        )
    else:
        filt.append("[voice]alimiter=limit=0.95,loudnorm=I=-16:TP=-1.5:LRA=11[afinal]")
    return ";".join(filt)

def build_long_video(proj, sentences, twords, cfg, out_path, ass_path, voiceover, music, audio_t0):
    print("[3/5] Building long video (ken burns + transitions + captions)...")
    n = len(sentences)
    is_section = [bool(re.match(cfg["section_pattern"], s["line"], flags=re.I))
                  for s in sentences]
    xf, sxf = cfg["xfade_duration"], cfg["section_xfade_duration"]
    # display durations: start-to-start; last sentence gets a tail
    D = [max(1.0, sentences[i + 1]["start"] - sentences[i]["start"]) for i in range(n - 1)]
    D.append(max(1.0, sentences[-1]["end"] - sentences[-1]["start"] + 1.0))
    dx = [sxf if is_section[i + 1] else xf for i in range(n - 1)]
    total_dur = sum(D)
    has_music = music is not None and music.exists()
    filt = build_filter(sentences, cfg, total_dur, has_music, D, dx, is_section, audio_t0,
                        ass_path=ass_path)
    cmd = ["ffmpeg", "-y"]
    for img in proj["images"]:
        cmd += ["-i", str(img)]
    cmd += ["-i", str(voiceover)]
    if has_music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
    cmd += ["-filter_complex", filt,
            "-map", "[vout]", "-map", "[afinal]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-r", str(FPS), "-c:a", "aac", "-b:a", "192k",
            "-t", f"{total_dur:.2f}", "-movflags", "+faststart",
            str(out_path)]
    run(cmd)
    print(f"      done -> {out_path} ({total_dur:.0f}s)")
    return total_dur

# ---------------------------------------------------------------- shorts
def build_shorts(proj, sentences, cfg, out_dir, voiceover, audio_t0):
    # Rendered fresh from source images in 9:16 (not cropped from the long
    # video) so captions and framing are clean with no duplicates.
    print("[4/5] Rendering Shorts (9:16)...")
    made = []
    for idx, (a, b) in enumerate(cfg.get("shorts", []), start=1):
        a = max(1, a) - 1
        b = min(len(sentences), b) - 1
        if b < a:
            continue
        seg = sentences[a:b + 1]
        s = seg[0]["start"]
        e = min(seg[-1]["end"], s + cfg["short_max_seconds"])
        rel = []
        for k, x in enumerate(seg):
            caps = [(t, a2 - s, b2 - s) for (t, a2, b2) in x["caps"] if b2 > s and a2 < e]
            if not caps:
                continue
            rel.append({"line": x["line"], "img": a + k,
                        "start": x["start"] - s, "end": min(x["end"], e) - s,
                        "caps": caps})
        if not rel:
            continue
        n = len(rel)
        is_section = [bool(re.match(cfg["section_pattern"], x["line"], flags=re.I)) for x in rel]
        xf, sxf = cfg["xfade_duration"], cfg["section_xfade_duration"]
        D = [max(1.0, rel[i + 1]["start"] - rel[i]["start"]) for i in range(n - 1)]
        D.append(max(1.0, rel[-1]["end"] - rel[-1]["start"] + 0.5))
        dx = [sxf if is_section[i + 1] else xf for i in range(n - 1)]
        total = sum(D)
        ass = out_dir / f"short_{idx}.ass"
        write_ass(ass, rel, SHORT_W, SHORT_H, 76, 340)
        filt = build_filter(rel, cfg, total, False, D, dx, is_section, s + audio_t0,
                            ow=SHORT_W, oh=SHORT_H, ass_path=ass)
        cmd = ["ffmpeg", "-y"]
        for x in rel:
            cmd += ["-i", str(proj["images"][x["img"]])]
        cmd += ["-i", str(voiceover)]
        out = out_dir / f"short_{idx}.mp4"
        cmd += ["-filter_complex", filt,
                "-map", "[vout]", "-map", "[afinal]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-r", str(FPS), "-c:a", "aac", "-b:a", "160k",
                "-t", f"{total:.2f}", "-movflags", "+faststart", str(out)]
        run(cmd)
        made.append(out)
        print(f"      short_{idx}.mp4 ({total:.0f}s, sentences {a + 1}-{a + n})")
    return made

# ---------------------------------------------------------------- main
def load_project(proj_dir):
    proj = Path(proj_dir)
    script = proj / "script.txt"
    if not script.exists():
        raise SystemExit("script.txt not found in project dir")
    lines = [l.strip() for l in script.read_text(encoding="utf-8").splitlines() if l.strip()]
    voice = None
    for ext in ("mp3", "wav", "m4a"):
        c = proj / f"voiceover.{ext}"
        if c.exists():
            voice = c
            break
    if not voice:
        raise SystemExit("voiceover.mp3 (or .wav/.m4a) not found in project dir")
    imgdir = proj / "images"
    if not imgdir.is_dir():
        raise SystemExit("images/ folder not found in project dir")
    images = sorted([p for p in imgdir.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    if len(images) != len(lines):
        raise SystemExit(f"images ({len(images)}) != script lines ({len(lines)}). "
                         "Name them 01..N in order.")
    music = proj / "music.mp3"
    if not music.exists():
        music = None
    return {"dir": proj, "lines": lines, "voice": voice, "images": images, "music": music}

def main():
    ap = argparse.ArgumentParser(description="The Extinct Files — auto edit tool")
    ap.add_argument("--project", required=True, help="project folder")
    ap.add_argument("--model", default="small", help="whisper model (tiny/base/small/medium)")
    ap.add_argument("--no-shorts", action="store_true")
    args = ap.parse_args()

    tool_dir = Path(__file__).parent
    cfg = dict(DEFAULT_CONFIG)
    for p in (tool_dir / "config.json", Path(args.project) / "config.json"):
        if p.exists():
            cfg.update(json.loads(p.read_text()))
            print(f"config loaded: {p}")

    proj = load_project(args.project)
    out_dir = proj["dir"] / "output"
    out_dir.mkdir(exist_ok=True)
    print(f"Project: {proj['dir'].name} | {len(proj['lines'])} sentences, "
          f"{len(proj['images'])} images, music={'yes' if proj['music'] else 'no'}")

    twords = transcribe_words(proj["voice"], args.model)
    sentences, twords = align_sentences(proj["lines"], twords)
    # shift timeline so video time == audio time (trims leading silence)
    audio_t0 = sentences[0]["start"]
    for s in sentences:
        s["start"] -= audio_t0
        s["end"] -= audio_t0
    for w in twords:
        w["start"] -= audio_t0
        w["end"] -= audio_t0
    build_caption_words(sentences, twords)

    ass_path = out_dir / "captions.ass"
    write_ass(ass_path, sentences, W, H, cfg["caption_fontsize"], 70)
    print("[3/5] captions written")

    long_out = out_dir / "long_final.mp4"
    build_long_video(proj, sentences, twords, cfg, long_out, ass_path,
                     proj["voice"], proj["music"], audio_t0)

    if not args.no_shorts:
        build_shorts(proj, sentences, cfg, out_dir, proj["voice"], audio_t0)
    print("[5/5] ALL DONE")
    print(f"  long : {long_out}")
    for f in sorted(out_dir.glob("short_*.mp4")):
        print(f"  short: {f}")

if __name__ == "__main__":
    main()

# The Extinct Files — Auto Edit Tool

Tum script, voiceover aur images do — tool khud **final long video + 3 Shorts** bana dega.

## Project folder ka structure

```
myvideo/
  script.txt        ← ek sentence per line, images ke order mein
  voiceover.mp3     ← poori narration ki audio
  images/           ← 01.jpg, 02.jpg, ... (jitni lines, utni images, order mein)
  music.mp3         ← optional background music
  config.json       ← optional (shorts ke sentence numbers yahan badlo)
```

## Chalane ka tareeqa

```bash
cd ~/workspace/extinct-files-tool
.venv/bin/python auto_edit.py --project /path/to/myvideo
```

Output `myvideo/output/` mein:
- `long_final.mp4` — 1920×1080, captions + name cards + music ducking
- `short_1.mp4`, `short_2.mp4`, `short_3.mp4` — 1080×1920 vertical Shorts

## Tool kya karta hai (auto)

1. **Timing** — Whisper se har sentence ka exact start/end nikalta hai
2. **Sync** — har image apne sentence par lagti hai
3. **Motion** — slow zoom-in/out, pan (Ken Burns), crossfade transitions
4. **Sections** — "Number five:" jaisi lines par dip-to-black + bada name card (DUNKLEOSTEUS)
5. **Captions** — karaoke-style, word-by-word highlight
6. **Audio** — background music khud halki hoti hai jab voiceover bole
7. **Shorts** — best moments se 30–40 sec ke 3 vertical clips

## Shorts ke moments badalna

`config.json` mein (ya project folder mein apni `config.json`):

```json
"shorts": [[1, 4], [16, 19], [37, 40]]
```

Numbers = script ki line numbers (1 se shuru). Har video ke liye alag set kar sakte ho.

## Notes

- Pehli baar Whisper model download hoga (~75MB for `tiny`), uske baad offline chalega.
- `--model tiny` tez hai (default), `--model small` ya `medium` zyada accurate.
- `--no-shorts` sirf long video banata hai.
- Zaroorat: Python 3.10+, `pip install faster-whisper`, aur `ffmpeg` installed ho.

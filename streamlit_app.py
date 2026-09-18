"""The Extinct Files — Auto Edit Tool (Streamlit edition).

Upload script + voiceover + images, get back the finished long video + 3 Shorts.
Deploy: Streamlit Community Cloud (free) — see README.md.
"""
import shutil
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))  # auto_edit.py lives next to this file

import streamlit as st

import auto_edit

# Bundled DejaVu Bold for name cards + captions
auto_edit.FONT_BOLD = str(APP_DIR / "fonts" / "DejaVuSans-Bold.ttf")
auto_edit.FONTS_DIR = str(APP_DIR / "fonts")

st.set_page_config(page_title="The Extinct Files — Auto Edit", layout="wide")
st.title("🎬 The Extinct Files — Auto Edit")
st.write("Script + voiceover + images do — final long video **aur** 3 vertical Shorts wapas lo.")


def parse_shorts(text):
    shorts = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(",")
        shorts.append([int(a), int(b)])
    return shorts


with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        script_text = st.text_area(
            "Script (ek narration sentence per line)",
            height=220,
            placeholder="This ocean looks empty. It isn't.\nMillions of years ago...",
        )
        voice_file = st.file_uploader("Voiceover (mp3/wav)", type=["mp3", "wav"])
        music_file = st.file_uploader("Background music (optional)", type=["mp3", "wav"])
    with col2:
        image_files = st.file_uploader(
            "Images (order mein — 01, 02, 03... jitni lines, utni images)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
        shorts_text = st.text_input(
            "Shorts (script line numbers: start,end; ...)",
            value="1,4; 16,19; 37,40",
        )
        model = st.selectbox("Whisper model", ["tiny", "small"], index=0)
    run_btn = st.form_submit_button("🚀 Video banao", type="primary")

st.caption("⏱️ Pehli baar Whisper model download hoga (~75MB), phir tez chalega. "
           "Lambi video ka render kuch minute le sakta hai.")


def run_pipeline(script_text, voice_file, image_files, music_file, shorts_text, model):
    # ---- validate
    lines = [l.strip() for l in script_text.splitlines() if l.strip()]
    if not lines:
        st.error("Script khaali hai — narration lines likho (ek sentence per line).")
        return
    if voice_file is None:
        st.error("Voiceover (mp3/wav) upload karo.")
        return
    if not image_files:
        st.error("Images upload karo (jitni script lines, utni images).")
        return
    if len(image_files) != len(lines):
        st.error(f"Script mein {len(lines)} lines hain, images {len(image_files)} hain — "
                 "dono barabar hone chahiye.")
        return
    try:
        shorts = parse_shorts(shorts_text)
    except Exception:
        st.error('Shorts ka format: "1,4; 16,19; 37,40" (line numbers, 1 se shuru).')
        return

    # ---- stage a project folder
    tmp = Path(tempfile.mkdtemp(prefix="extinct_"))
    (tmp / "images").mkdir()
    (tmp / "script.txt").write_text("\n".join(lines), encoding="utf-8")
    (tmp / "voiceover.mp3").write_bytes(voice_file.getvalue())
    ordered = sorted(image_files, key=lambda f: f.name)
    for i, f in enumerate(ordered):
        suffix = Path(f.name).suffix.lower() or ".jpg"
        (tmp / "images" / f"{i + 1:02d}{suffix}").write_bytes(f.getvalue())
    music = None
    if music_file is not None:
        music = tmp / "music.mp3"
        music.write_bytes(music_file.getvalue())

    cfg = dict(auto_edit.DEFAULT_CONFIG)
    cfg["shorts"] = shorts
    proj = auto_edit.load_project(tmp)
    out_dir = tmp / "output"
    out_dir.mkdir()

    bar = st.progress(0, text="Shuru ho raha hai...")

    # ---- pipeline
    with st.status("Voiceover sun raha hun (Whisper)...", expanded=False) as s:
        bar.progress(5, text="Voiceover sun raha hun (Whisper)...")
        twords = auto_edit.transcribe_words(proj["voice"], model)
        s.update(label="Voiceover samajh li ✅", state="complete")

    with st.status("Sentences ko voice se match kar raha hun...", expanded=False) as s:
        bar.progress(35, text="Sentences ko voice se match kar raha hun...")
        sentences, twords = auto_edit.align_sentences(proj["lines"], twords)
        audio_t0 = sentences[0]["start"]
        for x in sentences:
            x["start"] -= audio_t0
            x["end"] -= audio_t0
        for w in twords:
            w["start"] -= audio_t0
            w["end"] -= audio_t0
        auto_edit.build_caption_words(sentences, twords)
        s.update(label="Sentences match ho gaye ✅", state="complete")

    ass_path = out_dir / "captions.ass"
    auto_edit.write_ass(ass_path, sentences, auto_edit.W, auto_edit.H,
                        cfg["caption_fontsize"], 70)

    with st.status("Long video render ho rahi hai...", expanded=True) as s:
        bar.progress(50, text="Long video render ho rahi hai...")
        long_out = out_dir / "long_final.mp4"
        auto_edit.build_long_video(proj, sentences, twords, cfg, long_out,
                                   ass_path, proj["voice"], music, audio_t0)
        s.update(label="Long video tayyar ✅", state="complete")

    with st.status("Shorts render ho rahe hain...", expanded=False) as s:
        bar.progress(85, text="Shorts render ho rahe hain...")
        shorts_made = auto_edit.build_shorts(proj, sentences, cfg, out_dir,
                                             proj["voice"], audio_t0)
        s.update(label=f"{len(shorts_made)} Shorts tayyar ✅", state="complete")

    bar.progress(100, text="Ho gaya! ✅")
    return long_out, shorts_made


if run_btn:
    result = run_pipeline(script_text, voice_file, image_files,
                          music_file, shorts_text, model)
    if result:
        long_out, shorts_made = result
        st.success("Ho gaya! Neeche videos dekho aur download karo. ✅")
        st.subheader("Long video (16:9)")
        st.video(str(long_out))
        with open(long_out, "rb") as f:
            st.download_button("⬇️ Long video download karo",
                               f, file_name="long_final.mp4",
                               mime="video/mp4")
        st.subheader("Shorts (9:16)")
        cols = st.columns(3)
        for i, sp in enumerate(shorts_made):
            with cols[i % 3]:
                st.video(str(sp))
                with open(sp, "rb") as f:
                    st.download_button(f"⬇️ Short {i + 1} download karo",
                                       f, file_name=f"short_{i + 1}.mp4",
                                       mime="video/mp4",
                                       key=f"dl_short_{i}")

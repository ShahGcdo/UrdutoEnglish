import streamlit as st
from TTS.api import TTS
import tempfile
from moviepy.editor import TextClip, CompositeVideoClip, AudioFileClip, ColorClip
import os

st.set_page_config(page_title="🎙️ Text to Audio & Video", layout="centered")
st.title("🎙️ English Text ➝ Audio / Video Generator")

@st.cache_resource
def load_tts():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

tts = load_tts()

text = st.text_area("✍️ Enter English text:", height=150)

col1, col2 = st.columns(2)

with col1:
    if st.button("🔊 Generate Audio Only"):
        if text.strip():
            with st.spinner("Generating audio..."):
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tts.tts_to_file(text=text, file_path=temp_audio.name)

                with open(temp_audio.name, "rb") as f:
                    audio_bytes = f.read()
                    st.session_state.generated_audio = audio_bytes

with col2:
    if st.button("🎬 Generate Video (1280x720)"):
        if text.strip():
            with st.spinner("Generating audio and video..."):
                # Save audio first
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tts.tts_to_file(text=text, file_path=temp_audio.name)
                audio_clip = AudioFileClip(temp_audio.name)

                # Create black video background
                bg = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=audio_clip.duration)

                # Overlay text
                txt_clip = TextClip(text, fontsize=48, color='white', method='caption', size=(1200, None)).set_duration(audio_clip.duration).set_position("center")

                final_video = CompositeVideoClip([bg, txt_clip]).set_audio(audio_clip)
                output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
                final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

                with open(output_path, "rb") as f:
                    st.session_state.generated_video = f.read()

# Audio output
if "generated_audio" in st.session_state:
    st.audio(st.session_state.generated_audio, format="audio/wav")
    st.download_button("⬇️ Download MP3", st.session_state.generated_audio, file_name="english_tts.wav", mime="audio/wav")

# Video output
if "gener

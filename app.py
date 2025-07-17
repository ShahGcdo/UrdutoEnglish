import streamlit as st
from TTS.api import TTS
import io
import tempfile
from moviepy.editor import TextClip, CompositeVideoClip, AudioFileClip, ColorClip
import os

st.set_page_config(page_title="🎙️ Text to Audio & Video", layout="centered")
st.title("🎙️ English Text ➝ Audio / Video Generator")

@st.cache_resource
def load_tts():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

tts = load_tts()

# Text input
text = st.text_area("✍️ Enter English text:", height=150)

col1, col2 = st.columns(2)

with col1:
    if st.button("🔊 Generate Audio Only"):
        if text.strip():
            with st.spinner("Generating audio..."):
                audio_data = tts.tts(text)
                audio_bytes = io.BytesIO()
                tts.save_wav(audio_data, audio_bytes)
                audio_bytes.seek(0)
                st.session_state.generated_audio = audio_bytes.read()

with col2:
    if st.button("🎬 Generate Video (1280x720)"):
        if text.strip():
            with st.spinner("Generating audio and video..."):
                audio_data = tts.tts(text)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    tts.save_wav(audio_data, temp_audio_file)
                    temp_audio_file_path = temp_audio_file.name

                # Create black background
                background = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=AudioFileClip(temp_audio_file_path).duration)

                # Text overlay
                text_clip = TextClip(text, fontsize=48, color='white', size=(1200, None), method='caption', align='center')
                text_clip = text_clip.set_duration(background.duration).set_position('center')

                # Final video
                final = CompositeVideoClip([background, text_clip])
                final = final.set_audio(AudioFileClip(temp_audio_file_path))

                video_path = os.path.join(tempfile.gettempdir(), "generated_video.mp4")
                final.write_videofile(video_path, fps=24, codec="libx264", audio_codec="aac")

                with open(video_path, "rb") as file:
                    st.session_state.generated_video = file.read()

# Audio Output
if "generated_audio" in st.session_state:
    st.audio(st.session_state.generated_audio, format="audio/mp3")
    st.download_button("⬇️ Download MP3", st.session_state.generated_audio, file_name="english_tts.mp3", mime="audio/mp3")

# Video Output
if "generated_video" in st.session_state:
    st.video(st.session_state.generated_video)
    st.download_button("⬇️ Download Video", st.session_state.generated_video, file_name="text_to_video.mp4", mime="video/mp4")

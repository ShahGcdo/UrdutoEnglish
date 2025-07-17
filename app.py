import streamlit as st
from TTS.api import TTS
import tempfile
import os
from moviepy.editor import *
from datetime import datetime

st.set_page_config(page_title="Text to Audio/Video Generator", layout="centered")
st.title("🗣️ English Text to Audio & Video Generator")

# Load English TTS model
@st.cache_resource
def load_tts_model():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

tts = load_tts_model()

# Text input
text_input = st.text_area("✍️ Enter English Text", height=200)

# Select feature
feature = st.radio("Choose Output Type", ["🔊 Generate Audio", "🎬 Generate Video (1280x720)"])

if st.button("Generate"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating..."):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_audio_path = f"output_{timestamp}.wav"
            tts.tts_to_file(text=text_input, file_path=temp_audio_path)

            with open(temp_audio_path, "rb") as f:
                audio_data = f.read()
                st.session_state.generated_audio = audio_data

            if feature == "🎬 Generate Video (1280x720)":
                # Generate a simple text-on-screen video with audio
                temp_video_path = f"video_{timestamp}.mp4"

                txt_clip = TextClip(text_input, fontsize=40, color='white', size=(1280, 720), method='caption', bg_color='black')
                txt_clip = txt_clip.set_duration(AudioFileClip(temp_audio_path).duration)

                final_clip = txt_clip.set_audio(AudioFileClip(temp_audio_path))
                final_clip.write_videofile(temp_video_path, fps=24, codec='libx264', audio_codec='aac')

                with open(temp_video_path, "rb") as vf:
                    video_data = vf.read()
                    st.session_state.generated_video = video_data

                os.remove(temp_video_path)

            os.remove(temp_audio_path)

        st.success("✅ Generation completed!")

# Audio output
if "generated_audio" in st.session_state:
    st.audio(st.session_state.generated_audio, format="audio/wav")
    st.download_button("⬇️ Download MP3", st.session_state.generated_audio, file_name="english_tts.wav", mime="audio/wav")

# Video output
if "generated_video" in st.session_state:
    st.video(st.session_state.generated_video)
    st.download_button("⬇️ Download MP4", st.session_state.generated_video, file_name="text_to_video.mp4", mime="video/mp4")
mport streamlit as st
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

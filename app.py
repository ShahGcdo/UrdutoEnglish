import streamlit as st
from TTS.api import TTS
import tempfile
import os

st.set_page_config(page_title="🗣️ English Text to Speech", layout="centered")
st.title("🗣️ English Text to Speech")

@st.cache_resource
def load_tts():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

tts = load_tts()

text_input = st.text_area("Enter English Text", height=150)

if st.button("🔊 Generate Audio") and text_input.strip():
    with st.spinner("Synthesizing speech..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tts.tts_to_file(text=text_input, file_path=tmp_wav.name)
            audio_path = tmp_wav.name

        # Convert to MP3
        mp3_path = audio_path.replace(".wav", ".mp3")
        os.system(f"ffmpeg -y -i {audio_path} -codec:a libmp3lame -qscale:a 4 {mp3_path}")

        # Play audio
        st.audio(mp3_path, format="audio/mp3")

        # Download button
        with open(mp3_path, "rb") as f:
            st.download_button(label="⬇️ Download MP3", data=f, file_name="english_speech.mp3", mime="audio/mp3")

        os.remove(audio_path)
        os.remove(mp3_path)

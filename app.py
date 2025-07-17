import streamlit as st
from TTS.api import TTS
import io

st.set_page_config(page_title="🎙️ English Text-to-Speech", layout="centered")
st.title("🎙️ English Text ➝ Audio Generator")

# Load TTS model (English only)
@st.cache_resource
def load_tts():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

tts = load_tts()

# Input text box
text = st.text_area("✍️ Enter English text:", height=150)

# Speak button
if st.button("🔊 Generate Audio"):
    if text.strip():
        # Generate TTS
        with st.spinner("Generating speech..."):
            audio_data = tts.tts(text, speaker=tts.speakers[0], language=tts.languages[0])
            audio_bytes = io.BytesIO()
            tts.save_wav(audio_data, audio_bytes)
            audio_bytes.seek(0)
            st.session_state.generated_audio = audio_bytes.read()

# Show audio and download if available
if "generated_audio" in st.session_state:
    st.audio(st.session_state.generated_audio, format="audio/mp3")
    st.download_button(
        "⬇️ Download MP3",
        data=st.session_state.generated_audio,
        file_name="english_tts.mp3",
        mime="audio/mp3"
    )

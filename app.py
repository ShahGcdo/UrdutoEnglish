import streamlit as st
import os
import tempfile
from TTS.api import TTS
from gtts import gTTS
from io import BytesIO

st.set_page_config(page_title="🎙️ Urdu-English TTS App", layout="wide")
st.title("🎙️ Urdu & English Text to Audio App")

# Load English-only TTS model (safe for YouTube)
@st.cache_resource
def load_english_tts():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

english_tts = load_english_tts()

# Function to synthesize speech and return audio buffer
def synthesize_with_coqui(text, tts_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tts_model.tts_to_file(text=text, file_path=tmp_wav.name)
        tmp_wav.seek(0)
        return tmp_wav.name

def synthesize_with_gtts(text, lang):
    tts = gTTS(text, lang=lang)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

st.markdown("---")
st.header("✍️ English Text ➝ 🔊 Spoken English")

english_text = st.text_area("Enter English text here:", height=100)

if st.button("🔊 Generate English Audio"):
    if english_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating English audio..."):
            audio_path = synthesize_with_coqui(english_text, english_tts)
            st.audio(audio_path, format="audio/wav")
            with open(audio_path, "rb") as f:
                st.download_button("📥 Download Audio", f, file_name="english_audio.wav")

st.markdown("---")
st.header("✍️ Urdu Text ➝ 🔊 Spoken Urdu")

urdu_text = st.text_area("اردو متن یہاں درج کریں:", height=100)

if st.button("🔊 Generate Urdu Audio"):
    if urdu_text.strip() == "":
        st.warning("براہ کرم کچھ اردو متن درج کریں۔")
    else:
        with st.spinner("اردو آڈیو تیار ہو رہا ہے..."):
            mp3_fp = synthesize_with_gtts(urdu_text, lang="ur")
            st.audio(mp3_fp, format="audio/mp3")
            st.download_button("📥 Download Audio", mp3_fp, file_name="urdu_audio.mp3")

st.markdown("---")
st.caption("Made for YouTube monetization — all audio TTS is copyright-safe ✅")

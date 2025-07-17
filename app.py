import streamlit as st
import tempfile
import os
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS

# Load English TTS model (safe for monetization)
@st.cache_resource
def load_tts_model():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Load translation model (Urdu → English)
@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-ur-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_urdu_to_english(urdu_text):
    tokenizer, model = load_translation_model()
    tokens = tokenizer.prepare_seq2seq_batch([urdu_text], return_tensors="pt")
    translation = model.generate(**tokens)
    english_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return english_text

st.set_page_config(page_title="Urdu ↔ English Audio Translator", layout="centered")
st.title("🎙️ Urdu ↔ English Audio Translator (YouTube Safe)")

tts_model = load_tts_model()

# ---- Feature 1: Urdu Audio → English Audio ----
st.header("🎧 Urdu Audio ➝ English Audio")

audio_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"], key="urdu_audio")
if audio_file:
    st.audio(audio_file, format="audio/mp3")

    if st.button("🚀 Translate and Generate English Audio"):
        with st.spinner("Transcribing Urdu audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                urdu_audio_path = tmp.name

            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe(urdu_audio_path, language="ur")
            urdu_text = result["text"]
            st.markdown("### 📝 Transcribed Urdu")
            st.write(urdu_text)

        with st.spinner("Translating to English..."):
            english_text = translate_urdu_to_english(urdu_text)
            st.markdown("### 🌐 English Translation")
            st.write(english_text)

        with st.spinner("Generating English Audio..."):
            english_audio_path = os.path.join(tempfile.gettempdir(), "english_audio.wav")
            tts_model.tts_to_file(text=english_text, file_path=english_audio_path)
            st.success("✅ English Audio Generated")
            st.audio(english_audio_path)

            with open(english_audio_path, "rb") as f:
                st.download_button("⬇️ Download English Audio", f, file_name="english_audio.wav")

# ---- Feature 2: Urdu Text ➝ ⚠️ Urdu Audio (DISABLED) ----
st.header("📝 Urdu Text ➝ ⚠️ Urdu Audio (Not Available)")
st.warning("Urdu TTS model not available. Feature temporarily disabled.")

# ---- Feature 3: English Text ➝ English Audio ----
st.header("📝 English Text ➝ English Audio")
english_input = st.text_area("Enter English Text")
if st.button("🎤 Convert to English Audio"):
    if english_input.strip():
        with st.spinner("Generating English Audio..."):
            english_audio_path2 = os.path.join(tempfile.gettempdir(), "english_from_text.wav")
            tts_model.tts_to_file(text=english_input, file_path=english_audio_path2)
            st.success("✅ English Audio Generated")
            st.audio(english_audio_path2)

            with open(english_audio_path2, "rb") as f:
                st.download_button("⬇️ Download English Audio", f, file_name="english_from_text.wav")
    else:
        st.warning("Please enter some English text.")

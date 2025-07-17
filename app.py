import streamlit as st
import tempfile
import os
import whisper
from gtts import gTTS
from pydub import AudioSegment
from transformers import MarianMTModel, MarianTokenizer

st.set_page_config(page_title="Urdu to English", layout="centered")
st.title("🎤 Urdu Audio to English Translation")

# Load Whisper model (transcription)
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# Load translator model (Urdu → English)
@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-ur-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Convert uploaded file to WAV (16kHz mono)
def convert_audio(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name

        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            return tmp_wav.name
    except Exception as e:
        st.error(f"❌ Audio conversion failed: {e}")
        return None

# Translate Urdu text to English
def translate_text(urdu_text):
    tokenizer, model = load_translator()
    inputs = tokenizer(urdu_text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Generate English audio
def generate_audio(text, output_path):
    tts = gTTS(text, lang="en")
    tts.save(output_path)

# UI
uploaded_file = st.file_uploader("Upload Urdu Audio File", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("Translate and Generate English Audio"):
        with st.spinner("Processing..."):
            wav_path = convert_audio(uploaded_file)
            if not wav_path:
                st.stop()

            whisper_model = load_whisper()
            result = whisper_model.transcribe(wav_path, language="ur")
            urdu_text = result["text"]

            english_text = translate_text(urdu_text)

            out_path = wav_path.replace(".wav", "_en.mp3")
            generate_audio(english_text, out_path)

            st.markdown("### 📝 Urdu Transcript")
            st.write(urdu_text)

            st.markdown("### 🌍 English Translation")
            st.write(english_text)

            st.markdown("### 🔊 English Audio")
            with open(out_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")

            os.remove(wav_path)
            os.remove(out_path)

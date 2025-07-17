import streamlit as st
import torch
import tempfile
import os
import soundfile as sf
from transformers import pipeline, MarianMTModel, MarianTokenizer
from gtts import gTTS

st.set_page_config(page_title="Urdu to English Audio Translator", layout="centered")
st.title("🎙️ Urdu Audio to English Translation")

# ------------------------
# Model Selector UI
# ------------------------
st.markdown("## 🛠️ Choose Urdu ASR Model")
model_choice = st.selectbox("Select Urdu Transcription Model", [
    "🔊 Whisper-small (openai/whisper-small)",
    "🎧 Wav2Vec2 (kingabzpro/wav2vec2-large-xls-r-300m-Urdu)"
])

# ------------------------
# Model Loaders
# ------------------------
@st.cache_resource
def load_whisper_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=-1,  # CPU
        generation_kwargs={"language": "<|ur|>"}
    )

@st.cache_resource
def load_wav2vec2_model():
    return pipeline(
        "automatic-speech-recognition",
        model="kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
        device=-1  # CPU
    )

@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-ur-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# ------------------------
# Transcription Function
# ------------------------
def transcribe_urdu(audio_path, model_choice):
    if "Whisper" in model_choice:
        asr = load_whisper_model()
    else:
        asr = load_wav2vec2_model()
    result = asr(audio_path)
    return result["text"]

# ------------------------
# Translation Function
# ------------------------
def translate_urdu_to_english(urdu_text):
    tokenizer, model = load_translation_model()
    inputs = tokenizer(urdu_text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# ------------------------
# Text-to-Speech Function
# ------------------------
def generate_english_audio(text, output_path):
    tts = gTTS(text, lang="en")
    tts.save(output_path)

# ------------------------
# UI Upload and Process
# ------------------------
uploaded_file = st.file_uploader("📤 Upload Urdu Audio", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("✨ Translate and Generate English Audio"):
        with st.spinner("Processing..."):
            # Convert and save audio as clean WAV
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as tmp_raw:
                    tmp_raw.write(uploaded_file.read())
                    tmp_raw_path = tmp_raw.name

                audio_data, sample_rate = sf.read(tmp_raw_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    sf.write(tmp_wav.name, audio_data, sample_rate)
                    temp_audio_path = tmp_wav.name
                os.remove(tmp_raw_path)
            except Exception as e:
                st.error(f"❌ Audio conversion failed: {e}")
                st.stop()

            # Transcribe
            try:
                urdu_text = transcribe_urdu(temp_audio_path, model_choice)
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
                os.remove(temp_audio_path)
                st.stop()

            # Translate
            english_text = translate_urdu_to_english(urdu_text)

            # TTS
            english_audio_path = temp_audio_path.replace(".wav", "_english.mp3")
            generate_english_audio(english_text, english_audio_path)

        # Output
        st.markdown("### 📝 Transcribed Urdu Text")
        st.write(urdu_text)

        st.markdown("### 🌐 Translated English Text")
        st.write(english_text)

        st.success("✅ English Audio Generated!")
        with open(english_audio_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")

        os.remove(temp_audio_path)
        os.remove(english_audio_path)

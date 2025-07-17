import streamlit as st
import torch
import librosa
import tempfile
import os
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor, MarianMTModel, MarianTokenizer
from gtts import gTTS

st.set_page_config(page_title="Urdu to English Audio Translator", layout="centered")
st.title("🎙️ Urdu Audio to English Translation")

# Load Urdu ASR (Whisper-based)
@st.cache_resource
def load_urdu_asr():
    processor = WhisperProcessor.from_pretrained("sadnow/whisper-small-ur")
    model = WhisperForConditionalGeneration.from_pretrained("sadnow/whisper-small-ur")
    return processor, model

# Load translation model
@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-ur-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Transcribe Urdu
def transcribe_with_model(audio_path):
    processor, model = load_urdu_asr()
    audio, rate = librosa.load(audio_path, sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    generated_ids = model.generate(input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Translate Urdu to English
def translate_urdu_to_english(urdu_text):
    tokenizer, model = load_translation_model()
    inputs = tokenizer(urdu_text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Convert text to English speech
def generate_english_audio(text, output_path):
    tts = gTTS(text, lang="en")
    tts.save(output_path)

# UI
uploaded_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("✨ Translate and Generate English Audio"):
        with st.spinner("Processing..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                temp_audio_path = tmp.name

            # Transcribe Urdu
            try:
                urdu_text = transcribe_with_model(temp_audio_path)
            except Exception as e:
                st.error(f"Transcription failed: {str(e)}")
                os.remove(temp_audio_path)
                st.stop()

            # Translate
            english_text = translate_urdu_to_english(urdu_text)

            # Generate English Audio
            english_audio_path = temp_audio_path.replace(".wav", "_english.mp3")
            generate_english_audio(english_text, english_audio_path)

        # Show Results
        st.markdown("### 📝 Transcribed Urdu Text")
        st.write(urdu_text)

        st.markdown("### 🌐 Translated English Text")
        st.write(english_text)

        st.success("✅ English Audio Generated!")
        audio_file = open(english_audio_path, "rb")
        st.audio(audio_file.read(), format="audio/mp3")

        # Cleanup
        audio_file.close()
        os.remove(temp_audio_path)
        os.remove(english_audio_path)

import streamlit as st
import tempfile
import os
import gc
import torch
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS

# Caching large models
@st.cache_resource
def load_translation_model():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ur-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ur-en")
    return tokenizer, model

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")  # Lightweight for Streamlit Cloud

@st.cache_resource
def load_tts_models():
    urdu_tts = TTS(model_name="tts_models/ur/mai/tacotron2-DDC", progress_bar=False, gpu=False)
    english_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    return urdu_tts, english_tts

tokenizer, translation_model = load_translation_model()
urdu_tts, english_tts = load_tts_models()
whisper_model = load_whisper_model()

def translate_urdu_to_english(urdu_text):
    tokens = tokenizer.prepare_seq2seq_batch([urdu_text], return_tensors="pt")
    translation = translation_model.generate(**tokens)
    return tokenizer.decode(translation[0], skip_special_tokens=True)

# Streamlit UI
st.set_page_config(page_title="🎙️ Urdu-English Audio App", layout="centered")
st.title("🎙️ Urdu ↔ English Audio Translator")

tab1, tab2, tab3 = st.tabs(["🎧 Urdu Audio ➜ English Audio", "📝 Urdu Text ➜ Urdu Audio", "📝 English Text ➜ English Audio"])

# FEATURE 1: Urdu Audio ➜ Transcription ➜ English Translation ➜ English Audio
with tab1:
    audio_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"])

    if audio_file is not None:
        st.audio(audio_file, format="audio/mp3")
        if st.button("🚀 Translate and Generate English Audio"):
            with st.spinner("Transcribing Urdu audio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(audio_file.read())
                    temp_audio_path = tmp_file.name

                result = whisper_model.transcribe(temp_audio_path, language="ur")
                urdu_text = result["text"]
                st.markdown("### 📝 Transcribed Urdu Text")
                st.write(urdu_text)

            with st.spinner("Translating to English..."):
                english_text = translate_urdu_to_english(urdu_text)
                st.markdown("### 🌐 Translated English Text")
                st.write(english_text)

            with st.spinner("Generating English Audio..."):
                english_audio_path = os.path.join(tempfile.gettempdir(), "english_audio.wav")
                english_tts.tts_to_file(text=english_text, file_path=english_audio_path)

                st.success("✅ English Audio Generated!")
                st.audio(english_audio_path)
                with open(english_audio_path, "rb") as f:
                    st.download_button("⬇️ Download English Audio", f, "english_audio.wav")

            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()

# FEATURE 2: Urdu Text ➜ Urdu Audio
with tab2:
    urdu_text_input = st.text_area("✍️ Enter Urdu Text")
    if st.button("🎤 Generate Urdu Audio"):
        if urdu_text_input.strip() != "":
            with st.spinner("Generating Urdu audio..."):
                urdu_audio_path = os.path.join(tempfile.gettempdir(), "urdu_audio.wav")
                urdu_tts.tts_to_file(text=urdu_text_input, file_path=urdu_audio_path)

                st.success("✅ Urdu Audio Generated!")
                st.audio(urdu_audio_path)
                with open(urdu_audio_path, "rb") as f:
                    st.download_button("⬇️ Download Urdu Audio", f, "urdu_audio.wav")

            gc.collect()
            torch.cuda.empty_cache()

# FEATURE 3: English Text ➜ English Audio
with tab3:
    english_text_input = st.text_area("✍️ Enter English Text")
    if st.button("🗣️ Generate English Audio"):
        if english_text_input.strip() != "":
            with st.spinner("Generating English audio..."):
                english_text_path = os.path.join(tempfile.gettempdir(), "english_text_audio.wav")
                english_tts.tts_to_file(text=english_text_input, file_path=english_text_path)

                st.success("✅ English Audio Generated!")
                st.audio(english_text_path)
                with open(english_text_path, "rb") as f:
                    st.download_button("⬇️ Download English Audio", f, "english_text_audio.wav")

            gc.collect()
            torch.cuda.empty_cache()

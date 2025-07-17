import streamlit as st
import tempfile
import os
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS  # Coqui TTS
from pathlib import Path

# Initialize Coqui TTS (English and Urdu)
EN_TTS = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
UR_TTS = TTS(model_name="tts_models/ur/mai/tacotron2-DDC", progress_bar=False, gpu=False)

# Load translation model (Urdu to English)
MODEL_NAME = "Helsinki-NLP/opus-mt-ur-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

def translate_urdu_to_english(urdu_text):
    tokens = tokenizer.prepare_seq2seq_batch([urdu_text], return_tensors="pt")
    translation = model.generate(**tokens)
    english_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return english_text

def generate_tts_audio(text, lang='en'):
    tts_engine = EN_TTS if lang == 'en' else UR_TTS
    output_path = Path(tempfile.gettempdir()) / f"{lang}_speech.wav"
    tts_engine.tts_to_file(text=text, file_path=str(output_path))
    return str(output_path)

def convert_wav_to_mp3(wav_path):
    mp3_path = wav_path.replace(".wav", ".mp3")
    os.system(f"ffmpeg -y -i \"{wav_path}\" -ar 22050 \"{mp3_path}\"")
    return mp3_path

# Streamlit UI
st.set_page_config(page_title="Urdu-English Audio Translator", layout="centered")
st.title("🎙️ Urdu ↔ English Audio Features")

st.markdown("### 🎧 Feature 1: Urdu Audio ➜ English Audio")
audio_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/mp3")
    if st.button("🚀 Translate and Generate English Audio"):
        with st.spinner("Transcribing Urdu audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe(tmp_path, language="ur")
            urdu_text = result["text"]
            st.markdown("**📝 Urdu Transcription:**")
            st.write(urdu_text)

        with st.spinner("Translating to English..."):
            english_text = translate_urdu_to_english(urdu_text)
            st.markdown("**🌐 English Translation:**")
            st.write(english_text)

        with st.spinner("Generating English audio..."):
            wav_path = generate_tts_audio(english_text, lang='en')
            mp3_path = convert_wav_to_mp3(wav_path)
            st.audio(mp3_path, format="audio/mp3")
            with open(mp3_path, "rb") as f:
                st.download_button("⬇️ Download English Audio", f, file_name="english_audio.mp3")

# ---- Feature 2: Urdu Text ➜ Urdu Audio ----
st.markdown("---")
st.markdown("### 🗣️ Feature 2: Urdu Text ➜ Urdu Audio")
urdu_input = st.text_area("✍️ Enter Urdu Text Here")

if st.button("🎤 Generate Urdu Audio"):
    if urdu_input.strip() != "":
        with st.spinner("Generating Urdu speech..."):
            wav_path = generate_tts_audio(urdu_input, lang='ur')
            mp3_path = convert_wav_to_mp3(wav_path)
            st.audio(mp3_path, format="audio/mp3")
            with open(mp3_path, "rb") as f:
                st.download_button("⬇️ Download Urdu Audio", f, file_name="urdu_audio.mp3")
    else:
        st.warning("Please enter Urdu text.")

# ---- Feature 3: English Text ➜ English Audio ----
st.markdown("---")
st.markdown("### 🗣️ Feature 3: English Text ➜ English Audio")
english_input = st.text_area("✍️ Enter English Text Here")

if st.button("🎤 Generate English Audio"):
    if english_input.strip() != "":
        with st.spinner("Generating English speech..."):
            wav_path = generate_tts_audio(english_input, lang='en')
            mp3_path = convert_wav_to_mp3(wav_path)
            st.audio(mp3_path, format="audio/mp3")
            with open(mp3_path, "rb") as f:
                st.download_button("⬇️ Download English Audio", f, file_name="english_audio.mp3")
    else:
        st.warning("Please enter English text.")

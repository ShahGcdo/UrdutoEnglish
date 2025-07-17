import streamlit as st
import tempfile
import os
from pathlib import Path
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS

# Load TTS models
EN_TTS = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
UR_TTS = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

# Load translation model
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
    if lang == 'ur':
        tts_engine.tts_to_file(text=text, file_path=str(output_path), speaker="ur", language="ur")
    else:
        tts_engine.tts_to_file(text=text, file_path=str(output_path))
    return str(output_path)

# Streamlit UI
st.set_page_config(page_title="Urdu-English Audio Tools", layout="centered")
st.title("🎙️ Urdu ↔ English Audio Tools")

st.header("1️⃣ Urdu Audio ➡ Urdu Text ➡ English Text ➡ English Audio")

audio_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"])

if audio_file:
    st.audio(audio_file, format="audio/mp3")

    if st.button("🚀 Translate and Generate English Audio"):
        with st.spinner("Transcribing Urdu audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                temp_audio_path = tmp.name

            model_whisper = whisper.load_model("base")
            result = model_whisper.transcribe(temp_audio_path, language="ur")
            urdu_text = result["text"]

            st.markdown("### 📝 Transcribed Urdu Text")
            st.write(urdu_text)

        with st.spinner("Translating to English..."):
            english_text = translate_urdu_to_english(urdu_text)
            st.markdown("### 🌐 Translated English Text")
            st.write(english_text)

        with st.spinner("Generating English Audio..."):
            eng_audio_path = generate_tts_audio(english_text, lang='en')
            st.success("✅ English Audio Generated!")
            st.audio(eng_audio_path, format="audio/wav")
            with open(eng_audio_path, "rb") as f:
                st.download_button("⬇️ Download English Audio", f, file_name="english_audio.wav")

# Feature 2: Urdu Text ➡ Urdu Audio
st.header("2️⃣ Urdu Text ➡ Urdu Audio")

urdu_input = st.text_area("✍️ Enter Urdu Text")
if st.button("🔊 Generate Urdu Audio"):
    if urdu_input.strip():
        with st.spinner("Generating Urdu audio..."):
            urdu_audio_path = generate_tts_audio(urdu_input, lang='ur')
            st.success("✅ Urdu Audio Generated!")
            st.audio(urdu_audio_path, format="audio/wav")
            with open(urdu_audio_path, "rb") as f:
                st.download_button("⬇️ Download Urdu Audio", f, file_name="urdu_audio.wav")
    else:
        st.warning("⚠️ Please enter some Urdu text.")

# Feature 3: English Text ➡ English Audio
st.header("3️⃣ English Text ➡ English Audio")

english_input = st.text_area("✍️ Enter English Text")
if st.button("🔊 Generate English Audio"):
    if english_input.strip():
        with st.spinner("Generating English audio..."):
            eng_audio_path = generate_tts_audio(english_input, lang='en')
            st.success("✅ English Audio Generated!")
            st.audio(eng_audio_path, format="audio/wav")
            with open(eng_audio_path, "rb") as f:
                st.download_button("⬇️ Download English Audio", f, file_name="english_audio.wav")
    else:
        st.warning("⚠️ Please enter some English text.")

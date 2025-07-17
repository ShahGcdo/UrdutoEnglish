import streamlit as st
import tempfile
import os
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS

# Load Urdu-to-English translation model
MODEL_NAME = "Helsinki-NLP/opus-mt-ur-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

# Load Coqui TTS for English
EN_TTS = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Load Urdu TTS (multilingual model fallback)
try:
    UR_TTS = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
except Exception as e:
    st.warning("⚠️ Urdu TTS model failed to load. Urdu voice generation may not work.")
    UR_TTS = None

def translate_urdu_to_english(urdu_text):
    tokens = tokenizer.prepare_seq2seq_batch([urdu_text], return_tensors="pt")
    translation = model.generate(**tokens)
    english_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return english_text

# Streamlit app
st.set_page_config(page_title="Urdu ↔ English Audio App", layout="centered")
st.title("🎙️ Urdu ↔ English Audio & Translation App")

st.header("🎧 Feature 1: Urdu Audio → English Audio")

audio_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/mp3")

    if st.button("🚀 Translate and Generate English Audio"):
        with st.spinner("🔍 Transcribing Urdu audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.read())
                temp_audio_path = tmp_file.name

            model_whisper = whisper.load_model("base")
            result = model_whisper.transcribe(temp_audio_path, language="ur")
            urdu_text = result["text"]

            st.markdown("### 📝 Transcribed Urdu Text")
            st.write(urdu_text)

        with st.spinner("🌐 Translating to English..."):
            english_text = translate_urdu_to_english(urdu_text)

            st.markdown("### 🌍 Translated English Text")
            st.write(english_text)

        with st.spinner("🎤 Generating English Audio..."):
            output_path = os.path.join(tempfile.gettempdir(), "translated_audio_en.mp3")
            EN_TTS.tts_to_file(text=english_text, file_path=output_path)

            st.success("✅ English Audio Generated!")
            st.audio(output_path, format="audio/mp3")
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download English Audio", f, file_name="english_output.mp3")

# ------------------------------
st.markdown("---")
st.header("🗣️ Feature 2: Urdu Text → Urdu Audio")

urdu_input = st.text_area("✍️ Enter Urdu text")

if st.button("🔊 Generate Urdu Audio"):
    if UR_TTS is None:
        st.error("❌ Urdu TTS model not loaded. Cannot generate Urdu audio.")
    elif urdu_input.strip() == "":
        st.warning("⚠️ Please enter some Urdu text.")
    else:
        with st.spinner("🎤 Generating Urdu Audio..."):
            output_path_ur = os.path.join(tempfile.gettempdir(), "urdu_output.mp3")
            UR_TTS.tts_to_file(text=urdu_input, file_path=output_path_ur, speaker=UR_TTS.speakers[0])

            st.success("✅ Urdu Audio Generated!")
            st.audio(output_path_ur, format="audio/mp3")
            with open(output_path_ur, "rb") as f:
                st.download_button("⬇️ Download Urdu Audio", f, file_name="urdu_output.mp3")

# ------------------------------
st.markdown("---")
st.header("🗣️ Feature 3: English Text → English Audio")

english_input = st.text_area("✍️ Enter English text")

if st.button("🔊 Generate English Audio"):
    if english_input.strip() == "":
        st.warning("⚠️ Please enter some English text.")
    else:
        with st.spinner("🎤 Generating English Audio..."):
            output_path_en = os.path.join(tempfile.gettempdir(), "english_text_output.mp3")
            EN_TTS.tts_to_file(text=english_input, file_path=output_path_en)

            st.success("✅ English Audio Generated!")
            st.audio(output_path_en, format="audio/mp3")
            with open(output_path_en, "rb") as f:
                st.download_button("⬇️ Download English Audio", f, file_name="english_output.mp3")

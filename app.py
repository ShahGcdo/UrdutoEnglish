import streamlit as st
import transformers

st.write("✅ Transformers is installed!")

import streamlit as st
import tempfile
import os
from gtts import gTTS
import whisper
from transformers import MarianMTModel, MarianTokenizer

# Translation setup (Urdu to English)
MODEL_NAME = "Helsinki-NLP/opus-mt-ur-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

def translate_urdu_to_english(urdu_text):
    tokens = tokenizer.prepare_seq2seq_batch([urdu_text], return_tensors="pt")
    translation = model.generate(**tokens)
    english_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return english_text

# Streamlit app
st.set_page_config(page_title="Urdu to English Audio Translator", layout="centered")
st.title("🎙️ Urdu to English Audio Translator")

audio_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/mp3")

    if st.button("🚀 Translate and Generate English Audio"):
        with st.spinner("Transcribing Urdu audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.read())
                temp_audio_path = tmp_file.name

            model_whisper = whisper.load_model("base")
            result = model_whisper.transcribe(temp_audio_path, language="ur")
            urdu_text = result["text"]

            st.markdown("### 📝 Transcribed Urdu Text")
            st.write(urdu_text)

        with st.spinner("Translating to English..."):
            english_text = translate_urdu_to_english(urdu_text)

            st.markdown("### 🌐 Translated English Text")
            st.write(english_text)

        with st.spinner("Generating English audio..."):
            tts = gTTS(english_text, lang='en')
            output_path = os.path.join(tempfile.gettempdir(), "translated_audio.mp3")
            tts.save(output_path)

            st.success("✅ English Audio Generated!")
            st.audio(output_path, format="audio/mp3")

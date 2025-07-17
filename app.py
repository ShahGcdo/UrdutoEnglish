import streamlit as st
import tempfile
import os
from gtts import gTTS
import whisper
from googletrans import Translator

# Title
st.set_page_config(page_title="Urdu to English Audio Translator", layout="centered")
st.title("🎙️ Urdu to English Audio Translator")

# Upload audio file
audio_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/mp3")

    if st.button("🚀 Translate and Generate English Audio"):
        with st.spinner("Transcribing Urdu audio..."):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.read())
                temp_audio_path = tmp_file.name

            # Load Whisper model
            model = whisper.load_model("base")  # or "small" / "medium" for better quality
            result = model.transcribe(temp_audio_path, language="ur")
            urdu_text = result["text"]

            st.markdown("### 📝 Transcribed Urdu Text")
            st.write(urdu_text)

        with st.spinner("Translating to English..."):
            translator = Translator()
            translation = translator.translate(urdu_text, src='ur', dest='en')
            english_text = translation.text

            st.markdown("### 🌐 Translated English Text")
            st.write(english_text)

        with st.spinner("Generating English audio..."):
            tts = gTTS(english_text, lang='en')
            output_path = os.path.join(tempfile.gettempdir(), "translated_audio.mp3")
            tts.save(output_path)

            st.success("✅ English Audio Generated!")
            st.audio(output_path, format="audio/mp3")

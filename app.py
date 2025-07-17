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

# Feature 1: Urdu Audio ➝ Transcription ➝ English Translation ➝ English Audio
st.header("🔁 Urdu Audio ➝ English Audio")
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

            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download English Audio (MP3)",
                    data=f,
                    file_name="english_audio.mp3",
                    mime="audio/mpeg"
                )

# Feature 2: Urdu Text ➝ Urdu Audio
st.header("🗣️ Urdu Text to Urdu Audio")
urdu_input = st.text_area("✍️ Enter Urdu Text", key="urdu_text")

if st.button("🔊 Generate Urdu Audio"):
    if urdu_input.strip() != "":
        with st.spinner("Generating Urdu audio..."):
            tts = gTTS(urdu_input, lang='ur')
            urdu_audio_path = os.path.join(tempfile.gettempdir(), "urdu_audio.mp3")
            tts.save(urdu_audio_path)

            st.success("✅ Urdu Audio Generated!")
            st.audio(urdu_audio_path, format="audio/mp3")

            with open(urdu_audio_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Urdu Audio (MP3)",
                    data=f,
                    file_name="urdu_audio.mp3",
                    mime="audio/mpeg"
                )
    else:
        st.warning("⚠️ Please enter some Urdu text.")

# Feature 3: English Text ➝ English Audio
st.header("🗣️ English Text to English Audio")
english_input = st.text_area("✍️ Enter English Text", key="english_text")

if st.button("🔊 Generate English Audio"):
    if english_input.strip() != "":
        with st.spinner("Generating English audio..."):
            tts = gTTS(english_input, lang='en')
            english_audio_path = os.path.join(tempfile.gettempdir(), "english_audio.mp3")
            tts.save(english_audio_path)

            st.success("✅ English Audio Generated!")
            st.audio(english_audio_path, format="audio/mp3")

            with open(english_audio_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download English Audio (MP3)",
                    data=f,
                    file_name="english_audio.mp3",
                    mime="audio/mpeg"
                )
    else:
        st.warning("⚠️ Please enter some English text.")

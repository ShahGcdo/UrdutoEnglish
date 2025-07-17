import streamlit as st
import torch
import tempfile
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, MarianMTModel, MarianTokenizer
import whisper
from gtts import gTTS
from pydub import AudioSegment

st.set_page_config(page_title="Urdu to English Audio Translator", layout="centered")
st.title("🎙️ Urdu Audio to English Translation")

# Load ASR & translation models
@st.cache_resource
def load_wav2vec2():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    return processor, model

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-ur-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Convert uploaded audio to 16kHz mono WAV
def convert_audio(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as tmp_input:
            tmp_input.write(uploaded_file.read())
            tmp_input_path = tmp_input.name

        audio = AudioSegment.from_file(tmp_input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            return tmp_wav.name
    except Exception as e:
        st.error(f"❌ Audio conversion failed: {e}")
        return None

# Transcription
def transcribe_urdu(audio_path, method):
    if method == "wav2vec2":
        processor, model = load_wav2vec2()
        import librosa
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(pred_ids)[0]
    else:
        whisper_model = load_whisper()
        result = whisper_model.transcribe(audio_path, language="ur")
        return result["text"]

# Translation
def translate_urdu_to_english(urdu_text):
    tokenizer, model = load_translator()
    inputs = tokenizer(urdu_text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# English text to speech
def generate_english_audio(text, output_path):
    tts = gTTS(text, lang="en")
    tts.save(output_path)

# UI
uploaded_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"])
model_choice = st.selectbox("Choose Transcription Model", ["wav2vec2", "whisper"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("✨ Translate and Generate English Audio"):
        with st.spinner("Processing..."):
            temp_audio_path = convert_audio(uploaded_file)
            if not temp_audio_path:
                st.stop()

            try:
                urdu_text = transcribe_urdu(temp_audio_path, model_choice)
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
                os.remove(temp_audio_path)
                st.stop()

            english_text = translate_urdu_to_english(urdu_text)

            english_audio_path = temp_audio_path.replace(".wav", "_english.mp3")
            generate_english_audio(english_text, english_audio_path)

        st.markdown("### 📝 Transcribed Urdu Text")
        st.write(urdu_text)

        st.markdown("### 🌐 Translated English Text")
        st.write(english_text)

        st.success("✅ English Audio Generated!")
        with open(english_audio_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")

        os.remove(temp_audio_path)
        os.remove(english_audio_path)

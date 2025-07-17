import streamlit as st
import torch
import librosa
import tempfile
import os
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, MarianMTModel, MarianTokenizer
from faster_whisper import WhisperModel
from gtts import gTTS
from pydub import AudioSegment

st.set_page_config(page_title="Urdu to English Audio Translator", layout="centered")
st.title("🎙️ Urdu Audio to English Translation")

# Urdu ASR (wav2vec2)
@st.cache_resource
def load_urdu_asr():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    return processor, model

# Whisper (faster)
@st.cache_resource
def load_whisper():
    return WhisperModel("small", compute_type="int8")

# Translation model
@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-ur-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Transcribe Urdu with wav2vec2
def transcribe_with_wav2vec2(audio_path):
    processor, model = load_urdu_asr()
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

# Transcribe Urdu with Whisper
def transcribe_with_whisper(audio_path):
    model = load_whisper()
    segments, _ = model.transcribe(audio_path, language="ur")
    return " ".join([seg.text for seg in segments])

# Translate Urdu → English
def translate_urdu_to_english(urdu_text):
    tokenizer, model = load_translation_model()
    inputs = tokenizer(urdu_text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Generate English audio
def generate_english_audio(text, output_path):
    tts = gTTS(text, lang="en")
    tts.save(output_path)

# Convert to WAV if needed
def convert_to_wav(input_path, output_path):
    try:
        sound = AudioSegment.from_file(input_path)
        sound.export(output_path, format="wav")
        return output_path
    except Exception as e:
        st.error(f"❌ Audio conversion failed: {str(e)}")
        return None

# UI
uploaded_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a"])
transcriber_choice = st.selectbox("Choose Urdu Transcriber", ["🟢 Accurate (wav2vec2)", "⚪ Fast (Whisper)"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("✨ Translate and Generate English Audio"):
        with st.spinner("Processing..."):
            # Save and convert to WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                temp_input = tmp.name

            temp_wav = temp_input + ".wav"
            wav_path = convert_to_wav(temp_input, temp_wav)
            if not wav_path:
                st.stop()

            # Transcribe Urdu
            try:
                if "wav2vec2" in transcriber_choice.lower():
                    urdu_text = transcribe_with_wav2vec2(wav_path)
                else:
                    urdu_text = transcribe_with_whisper(wav_path)
            except Exception as e:
                st.error(f"❌ Transcription failed: {str(e)}")
                os.remove(wav_path)
                st.stop()

            # Translate
            english_text = translate_urdu_to_english(urdu_text)

            # Generate English Audio
            english_audio_path = wav_path.replace(".wav", "_english.mp3")
            generate_english_audio(english_text, english_audio_path)

        # Results
        st.markdown("### 📝 Transcribed Urdu Text")
        st.write(urdu_text)

        st.markdown("### 🌐 Translated English Text")
        st.write(english_text)

        st.success("✅ English Audio Generated!")
        with open(english_audio_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")

        # Cleanup
        os.remove(temp_input)
        os.remove(wav_path)
        os.remove(english_audio_path)

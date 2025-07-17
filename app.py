import streamlit as st
import tempfile
import os
import io
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, MarianMTModel, MarianTokenizer
from faster_whisper import WhisperModel
from gTTS import gTTS
from pydub import AudioSegment

st.set_page_config(page_title="Urdu to English Audio Translator", layout="centered")
st.title("🎙️ Urdu Audio to English Translation")

# ------------------------
# Model selector
# ------------------------
transcriber_choice = st.selectbox(
    "Choose Urdu Transcriber",
    ["🟢 Fast & Reliable (Whisper-small)", "🔬 Experimental (Wav2Vec2 Multilingual)"],
    help="Whisper-small usually gives better Urdu results on CPU. Wav2Vec2 multilingual is not fine-tuned for Urdu text and may produce poor transcripts."
)

# ------------------------
# Cached model loaders
# ------------------------
@st.cache_resource
def load_whisper_small():
    # int8 for low memory; falls back to CPU
    return WhisperModel("small", compute_type="int8")

@st.cache_resource
def load_wav2vec2_xlsr53():
    # NOTE: This model is not Urdu-fine-tuned; results may be noisy.
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    return processor, model

@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-ur-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# ------------------------
# Audio conversion
# ------------------------
def convert_to_wav_16k(file_bytes: bytes) -> str | None:
    """
    Convert uploaded audio bytes (mp3, m4a, wav, etc.) to a temp 16kHz mono WAV.
    Returns path to WAV file or None on failure.
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"❌ Could not read audio file: {e}")
        return None

    # force 16kHz mono for ASR
    audio = audio.set_channels(1).set_frame_rate(16000)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            return tmp_wav.name
    except Exception as e:
        st.error(f"❌ Failed to export WAV: {e}")
        return None

# ------------------------
# Transcribers
# ------------------------
def transcribe_whisper(audio_path: str) -> str:
    model = load_whisper_small()
    segments, _info = model.transcribe(audio_path, language="ur", task="transcribe")
    return " ".join(seg.text for seg in segments).strip()

def transcribe_wav2vec2(audio_path: str) -> str:
    # Warning: not Urdu-fine-tuned; may produce junk text.
    processor, model = load_wav2vec2_xlsr53()
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip()

# ------------------------
# Translation
# ------------------------
def translate_urdu_to_english(urdu_text: str) -> str:
    tokenizer, model = load_translator()
    if not urdu_text.strip():
        return ""
    inputs = tokenizer(urdu_text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True).strip()

# ------------------------
# English TTS
# ------------------------
def synthesize_english_audio(text: str) -> str | None:
    if not text.strip():
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
            tts = gTTS(text, lang="en")
            tts.save(tmp_mp3.name)
            return tmp_mp3.name
    except Exception as e:
        st.error(f"❌ English speech synthesis failed: {e}")
        return None

# ------------------------
# File upload UI
# ------------------------
uploaded_file = st.file_uploader("Upload Urdu Audio", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    # Read file to bytes once
    file_bytes = uploaded_file.read()

    # Play original audio
    st.audio(file_bytes, format="audio/mp3")

    if st.button("✨ Translate and Generate English Audio"):
        with st.spinner("Processing..."):

            # Convert to WAV
            wav_path = convert_to_wav_16k(file_bytes)
            if not wav_path:
                st.stop()

            # Transcribe
            try:
                if transcriber_choice.startswith("🟢"):
                    urdu_text = transcribe_whisper(wav_path)
                else:
                    urdu_text = transcribe_wav2vec2(wav_path)
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                st.stop()

            # Translate
            try:
                english_text = translate_urdu_to_english(urdu_text)
            except Exception as e:
                st.error(f"❌ Translation failed: {e}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                st.stop()

            # English speech
            mp3_path = synthesize_english_audio(english_text)

        # Results
        st.subheader("📝 Urdu Transcript")
        st.write(urdu_text if urdu_text else "_(empty)_")

        st.subheader("🌍 English Translation")
        st.write(english_text if english_text else "_(empty)_")

        if mp3_path:
            st.success("✅ English Audio Generated!")
            with open(mp3_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")

        # Cleanup
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if mp3_path and os.path.exists(mp3_path):
            os.remove(mp3_path)

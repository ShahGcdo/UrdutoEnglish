import streamlit as st
from TTS.api import TTS
from moviepy.editor import TextClip, CompositeVideoClip, AudioFileClip
import tempfile
import os
import time
from pydub import AudioSegment

st.set_page_config(page_title="🗣️ Text to Audio & Video", layout="centered")
st.title("🗣️ English Text to Audio & Video (1280x720)")

@st.cache_resource
def load_tts():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

tts = load_tts()

# Function to generate TTS audio
def generate_audio(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tts.tts_to_file(text=text, file_path=tmp_wav.name)
        return tmp_wav.name

# Convert WAV to MP3 for download
def convert_to_mp3(wav_path):
    sound = AudioSegment.from_wav(wav_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        sound.export(tmp_mp3.name, format="mp3")
        return tmp_mp3.name

# Generate text-based video with audio
def generate_video(text, audio_path):
    audioclip = AudioFileClip(audio_path)
    duration = audioclip.duration

    clip = TextClip(
        text,
        fontsize=44,
        color='white',
        size=(1280, 720),
        method='caption',
        bg_color='black',
        align='center'
    ).set_duration(duration)

    videoclip = clip.set_audio(audioclip)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        st.info("🎞️ Generating video... please wait ⏳")
        start = time.time()
        videoclip.write_videofile(
            tmp_video.name,
            fps=12,  # lower FPS for faster rendering
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            threads=2,
            logger=None
        )
        print(f"✅ Video generated in {time.time() - start:.2f} seconds.")
        return tmp_video.name

# UI Tabs
tab1, tab2 = st.tabs(["🔊 Text to Audio", "🎬 Text to Video"])

# Text to Audio
with tab1:
    eng_text = st.text_area("Enter English Text", height=150)

    if st.button("🔈 Generate Audio"):
        if eng_text.strip():
            audio_path = generate_audio(eng_text)
            st.session_state.generated_audio = convert_to_mp3(audio_path)
            st.audio(st.session_state.generated_audio, format="audio/mp3")
            st.download_button("⬇️ Download MP3", open(st.session_state.generated_audio, "rb"), file_name="speech.mp3", mime="audio/mp3")
        else:
            st.warning("⚠️ Please enter some text.")

# Text to Video
with tab2:
    eng_vid_text = st.text_area("Enter Text for Video (with voice)", height=150)

    if st.button("🎞️ Generate Video"):
        if eng_vid_text.strip():
            audio_path = generate_audio(eng_vid_text)
            video_path = generate_video(eng_vid_text, audio_path)
            st.session_state.generated_video = video_path
            st.video(video_path)
            st.download_button("⬇️ Download MP4", open(video_path, "rb"), file_name="text_video.mp4", mime="video/mp4")
        else:
            st.warning("⚠️ Please enter some text.")

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *
from TTS.api import TTS
import tempfile
import os

st.set_page_config(page_title="📖 Story Video with Voice", layout="wide")
st.title("📖 Story Narrator with Video and Voice")

# Load Coqui TTS model once
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

def generate_clip_with_audio(line_text, index):
    # Create image
    img = Image.new("RGB", (1280, 720), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)

    w, h = draw.textsize(line_text, font=font)
    draw.text(((1280 - w) / 2, (720 - h) / 2), line_text, font=font, fill="white")

    # Save image to temp file
    img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    img.save(img_path)

    # Generate audio using Coqui TTS
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    tts_model.tts_to_file(text=line_text, file_path=audio_path)

    # Load audio and image
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    img_clip = ImageClip(img_path).set_duration(duration)

    # Combine audio + video
    final_clip = img_clip.set_audio(audio_clip)

    return final_clip

def create_story_video(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    clips = [generate_clip_with_audio(line, idx) for idx, line in enumerate(lines)]
    final_video = concatenate_videoclips(clips, method="compose")

    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    final_video.write_videofile(output_path, fps=24)
    return output_path

# Streamlit UI
input_text = st.text_area("📜 Enter Story Text (each line will be narrated separately):", height=300)

if st.button("🎬 Generate Video with Voice"):
    with st.spinner("Generating..."):
        video_path = create_story_video(input_text)
        st.success("Done! Preview below 👇")
        st.video(video_path)
        with open(video_path, "rb") as file:
            st.download_button("⬇️ Download Video", file.read(), "story_video.mp4")

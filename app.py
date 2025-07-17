import streamlit as st
import os
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import tempfile
import uuid
from TTS.api import TTS

# Initialize TTS model (Coqui)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

st.set_page_config(page_title="📖 Text-to-Story Video", layout="centered")
st.title("📖 Create Story Video with Narration")

def generate_clip_with_audio(line_text, index):
    # Generate TTS audio
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, f"line_{index}.wav")
    tts.tts_to_file(text=line_text, file_path=audio_path)

    # Create an image with the text
    img = Image.new("RGB", (1280, 720), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 48)

    # Get text size using font.getsize
    w, h = font.getsize(line_text)
    draw.text(((1280 - w) / 2, (720 - h) / 2), line_text, font=font, fill="white")

    # Save image
    img_path = os.path.join(temp_dir, f"frame_{index}.png")
    img.save(img_path)

    # Create video clip
    audio_clip = AudioFileClip(audio_path)
    img_clip = ImageClip(img_path).set_duration(audio_clip.duration)
    img_clip = img_clip.set_audio(audio_clip)

    return img_clip

def create_story_video(full_text):
    lines = [line.strip() for line in full_text.strip().split("\n") if line.strip()]
    clips = [generate_clip_with_audio(line, idx) for idx, line in enumerate(lines)]

    final_video = concatenate_videoclips(clips, method="compose")
    output_path = os.path.join(tempfile.gettempdir(), f"story_{uuid.uuid4().hex}.mp4")
    final_video.write_videofile(output_path, fps=24)

    return output_path

# UI
input_text = st.text_area("✏️ Enter Story Text (one sentence per line):", height=250)

if st.button("🎬 Generate Video"):
    if input_text.strip():
        with st.spinner("Generating your narrated video..."):
            video_path = create_story_video(input_text)
            st.success("✅ Video generated successfully!")
            st.video(video_path)
    else:
        st.warning("Please enter some story text.")

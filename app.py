import streamlit as st
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from TTS.api import TTS

# Initialize Coqui TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

st.set_page_config(page_title="📖 Story Video Generator", layout="centered")
st.title("📖 Story to Video with Voice")

input_text = st.text_area("✍️ Enter Story Text", height=300)

if st.button("🎬 Generate Story Video"):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating video..."):
            def generate_clip_with_audio(line_text, index):
                # Temp directory
                temp_dir = tempfile.mkdtemp()

                # Generate TTS audio
                audio_path = os.path.join(temp_dir, f"line_{index}.wav")
                tts.tts_to_file(text=line_text, file_path=audio_path)

                # Create image with text
                img = Image.new("RGB", (1280, 720), color=(0, 0, 0))
                draw = ImageDraw.Draw(img)
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                font = ImageFont.truetype(font_path, 48)

                # Get text size using bbox
                bbox = draw.textbbox((0, 0), line_text, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                draw.text(((1280 - w) / 2, (720 - h) / 2), line_text, font=font, fill="white")

                # Save image
                img_path = os.path.join(temp_dir, f"frame_{index}.png")
                img.save(img_path)

                # Create video clip with audio
                audio_clip = AudioFileClip(audio_path)
                img_clip = ImageClip(img_path).set_duration(audio_clip.duration).set_audio(audio_clip)
                return img_clip

            # Process lines
            lines = [line.strip() for line in input_text.split("\n") if line.strip()]
            clips = [generate_clip_with_audio(line, idx) for idx, line in enumerate(lines)]

            final_video = concatenate_videoclips(clips, method="compose")

            # Save final video
            output_path = os.path.join(tempfile.gettempdir(), "story_video.mp4")
            final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

        st.success("✅ Video Generated Successfully!")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("📥 Download Video", data=f, file_name="story_video.mp4", mime="video/mp4")

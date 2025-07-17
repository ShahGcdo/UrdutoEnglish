import streamlit as st
from moviepy.editor import TextClip, CompositeVideoClip
import tempfile
import os

st.set_page_config(page_title="📝 Text to Scrolling Video", layout="centered")
st.title("📝 Text to Scrolling Video (No Audio)")

# Set ImageMagick path if needed
os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"

def generate_scrolling_video(text):
    lines = text.strip().split('\n')
    num_lines = len(lines)
    duration_per_line = 2.5  # seconds per line
    total_duration = duration_per_line * num_lines

    clips = []
    for i, line in enumerate(lines):
        txt_clip = TextClip(
            line,
            fontsize=48,
            color='white',
            size=(1280, 720),
            method='caption',
            bg_color='black'
        ).set_duration(duration_per_line).set_start(i * duration_per_line)
        clips.append(txt_clip)

    video = CompositeVideoClip(clips, size=(1280, 720)).set_duration(total_duration)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        video.write_videofile(tmp_video.name, fps=24, codec="libx264", audio=False)
        return tmp_video.name

# UI
input_text = st.text_area("Enter each line of text on a new line", height=200)

if st.button("🎞️ Generate Video"):
    if input_text.strip():
        video_path = generate_scrolling_video(input_text)
        st.video(video_path)
        st.download_button("⬇️ Download Video", open(video_path, "rb"), file_name="scrolling_text_video.mp4", mime="video/mp4")
    else:
        st.warning("Please enter text to generate video.")

import streamlit as st
import os
import tempfile
from moviepy.editor import TextClip, CompositeVideoClip, concatenate_videoclips, ColorClip

st.set_page_config(page_title="📝 Text to Video Generator", layout="centered")
st.title("📝 Text to Scrolling Video (1280x720)")

# Optional: fix ImageMagick path (you can comment this if not needed)
os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"  # Adjust if needed

def generate_scrolling_video(text):
    lines = text.strip().split('\n')
    duration_per_line = 3  # seconds
    clips = []

    for line in lines:
        if line.strip() == "":
            continue  # skip empty lines
        try:
            txt_clip = TextClip(
                line,
                fontsize=48,
                color='white',
                font='DejaVu-Sans',  # Use a common font to avoid missing font errors
                size=(1280, 720),
                method='label',
                bg_color='black'
            ).set_duration(duration_per_line)
            clips.append(txt_clip)
        except Exception as e:
            st.error(f"Error creating clip for line: {line}\n{e}")
            return None

    if not clips:
        st.warning("No valid text lines to generate video.")
        return None

    final_clip = concatenate_videoclips(clips, method="compose")

    # Save to temp file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    final_clip.write_videofile(temp_video.name, fps=24, codec='libx264', audio=False)

    return temp_video.name

# Streamlit UI
input_text = st.text_area("✍️ Enter text (each line will show separately):", height=300)

if st.button("🎬 Generate Video"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating video..."):
            video_path = generate_scrolling_video(input_text)
            if video_path:
                st.success("✅ Video generated successfully!")
                st.video(video_path)
                with open(video_path, "rb") as f:
                    st.download_button("⬇️ Download Video", f, file_name="text_video.mp4", mime="video/mp4")

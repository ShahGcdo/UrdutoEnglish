import streamlit as st
import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, concatenate_videoclips

st.set_page_config(page_title="📝 Text to Video Generator", layout="centered")
st.title("📝 Text to Scrolling Video (No ImageMagick)")

# Load a basic TTF font (included with PIL or from system)
def get_font(size=48):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()

def render_text_to_image(text, size=(1280, 720), font_size=48):
    img = Image.new("RGB", size, color="black")
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)

    # Wrap text if too long
    lines = []
    words = text.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        if draw.textlength(test_line, font=font) > size[0] - 100:
            lines.append(line)
            line = word + " "
        else:
            line = test_line
    lines.append(line)

    # Vertical centering
    y_text = (size[1] - (len(lines) * font_size)) // 2
    for l in lines:
        text_width = draw.textlength(l, font=font)
        x = (size[0] - text_width) // 2
        draw.text((x, y_text), l.strip(), font=font, fill="white")
        y_text += font_size + 10

    return img

def generate_video_from_text(text_lines):
    duration_per_slide = 3
    clips = []

    for line in text_lines:
        if line.strip() == "":
            continue
        image = render_text_to_image(line.strip())
        temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(temp_img.name)

        clip = ImageClip(temp_img.name).set_duration(duration_per_slide)
        clips.append(clip)

    if not clips:
        return None

    final_clip = concatenate_videoclips(clips, method="compose")
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    final_clip.write_videofile(temp_video.name, fps=24, codec="libx264", audio=False)
    return temp_video.name

# Streamlit UI
input_text = st.text_area("✍️ Enter your story (each line will show as a slide):", height=300)

if st.button("🎬 Generate Video"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating video..."):
            lines = input_text.strip().split('\n')
            video_path = generate_video_from_text(lines)
            if video_path:
                st.success("✅ Video generated!")
                st.video(video_path)
                with open(video_path, "rb") as f:
                    st.download_button("⬇️ Download Video", f, file_name="text_video.mp4", mime="video/mp4")
            else:
                st.error("❌ Failed to generate video.")

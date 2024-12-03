import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import subprocess

@st.cache_resource
def load_model(model_path):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

# Path to your trained model
MODEL_PATH = "models/accident_detection/model.h5"
model = load_model(MODEL_PATH)

def preprocess_frame(frame, img_height=250, img_width=250):
    """Preprocess the video frame for prediction."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (img_width, img_height))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

def predict_frame(frame):
    """Predict whether an accident is detected in the frame."""
    preprocessed = preprocess_frame(frame)
    prediction = model.predict(preprocessed)
    return "Accident Detected" if prediction[0][0] > 0.5 else "No Accident Detected"

def download_youtube_video(youtube_url, output_path="video.mp4"):
    """Download YouTube video using yt-dlp."""
    try:
        command = f"yt-dlp --quiet --no-warnings --merge-output-format mp4 -o {output_path} {youtube_url}"
        process = subprocess.run(command, shell=True, text=True, capture_output=True)

        if process.returncode != 0:
            raise Exception(f"yt-dlp failed with error: {process.stderr.strip()}")

        return output_path
    except Exception as e:
        raise Exception(f"Failed to download video: {e}")

def show_accident_detection():
    """Main function to detect accidents from a YouTube video."""
    st.title("Accident Detection from CCTV Footage")

    # Create two-column layout
    col_left, col_right = st.columns([2, 3])

    with col_left:
        # Input for YouTube video URL
        st.markdown("#### Upload YouTube Video")
        youtube_url = st.text_input("Paste the YouTube video URL below:")

        if st.button("Download and Analyze Video"):
            if not youtube_url:
                st.error("Please enter a valid YouTube URL.")
                return

            # Delete previous video file if exists
            if os.path.exists("video.mp4"):
                os.remove("video.mp4")

            with st.spinner("Downloading YouTube video..."):
                try:
                    video_path = download_youtube_video(youtube_url)
                    st.success("YouTube video downloaded successfully!")
                except Exception as e:
                    st.error(f"Failed to download video: {e}")
                    return

            # Display the video in the left column
            st.markdown("### Video Preview")
            st.video(video_path)

    with col_right:
        st.markdown("### Accident Analysis")
        st.info("Analyzing video frames for accidents...")

        if os.path.exists("video.mp4"):
            # Open the video file
            cap = cv2.VideoCapture("video.mp4")
            frame_count = 0
            accident_detected = False

            if not cap.isOpened():
                st.error("Unable to open video file!")
                return

            # Process video frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                frame_count += 1

                # Analyze every 30th frame for accident detection
                if frame_count % 30 == 0:
                    result = predict_frame(frame)
                    if result == "Accident Detected":
                        accident_detected = True
                        st.warning(f"Accident Detected at Frame {frame_count}")

                        # Display accident frame in the right column
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(
                            frame_rgb,
                            caption=f"Accident Detected at Frame {frame_count}",
                            use_container_width=True,
                        )

            cap.release()
            if not accident_detected:
                st.success("No accidents detected in the video.")
            else:
                st.info("Accident detection completed.")

        if st.button("Reset"):
            if os.path.exists("video.mp4"):
                os.remove("video.mp4")
            st.experimental_rerun()




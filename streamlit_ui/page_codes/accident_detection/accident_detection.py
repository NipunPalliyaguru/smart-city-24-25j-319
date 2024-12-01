import streamlit as st
import cv2
import os
import numpy as np
import tensorflow as tf
import subprocess

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

MODEL_PATH = "models/accident_detection/model.h5"
model = load_model(MODEL_PATH)

def preprocess_frame(frame, img_height=250, img_width=250):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (img_width, img_height))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

def predict_frame(frame):
    preprocessed = preprocess_frame(frame)
    prediction = model.predict(preprocessed)
    return "Accident Detected" if prediction[0][0] > 0.5 else "No Accident Detected"

def download_youtube_video(youtube_url, output_path="video.mp4"):
    try:
        # Use yt-dlp to download the video
        command = f"yt-dlp -f best -o {output_path} {youtube_url}"
        subprocess.run(command, shell=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to download video: {str(e)}")

def show_accident_detection():
    st.title("Accident Detection from Video")

    youtube_url = st.text_input("Enter YouTube Video URL:")
    if st.button("Process Video"):
        if not youtube_url:
            st.error("Please enter a valid YouTube URL.")
            return

        with st.spinner("Downloading video..."):
            try:
                video_path = download_youtube_video(youtube_url)
                st.success("Video downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download video: {e}")
                return

        st.info("Processing video frame by frame...")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        accident_detected = False

        if not cap.isOpened():
            st.error("Unable to open video file!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            frame_count += 1

            if frame_count % 30 == 0:
                result = predict_frame(frame)
                if result == "Accident Detected":
                    accident_detected = True
                    st.warning(f"Accident Detected at Frame {frame_count}")

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Accident Detected at Frame {frame_count}", use_column_width=True)

        cap.release()
        if not accident_detected:
            st.success("No accidents detected in the video.")
        else:
            st.info("Accident detection completed.")


import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model_path = "models/waste_management/hybrid_resnet50_effb1_classifier.h5"
labels_path = "models/waste_management/labels.txt"
model = load_model(model_path, compile=False)

# Load the labels
# class_names = [label.strip() for label in open(labels_path, "r").readlines()]
class_names = class_names = [label.split(" ", 1)[1].strip() for label in open(labels_path, "r").readlines()]

# Bin images mapping
bin_images = {
    "biological": {
        "closed": "page_codes/waste_management/assets/green_bin_lid_closed.jpg",
        "open": "page_codes/waste_management/assets/green_bin_lid_opened.jpg",
    },
    "glass": {
        "closed": "page_codes/waste_management/assets/orange_bin_lid_closed.jpg",
        "open": "page_codes/waste_management/assets/orange_bin_lid_opened.jpg",
    },
    "paper": {
        "closed": "page_codes/waste_management/assets/yellow_bin_lid_closed.jpg",
        "open": "page_codes/waste_management/assets/yellow_bin_lid_opened.jpg",
    },
    "plastic": {
        "closed": "page_codes/waste_management/assets/red_bin_lid_closed.jpg",
        "open": "page_codes/waste_management/assets/red_bin_lid_opened.jpg",
    },
}


# Function to classify uploaded image
def classify_image(image):
    # Ensure image is in RGB format
    image = image.convert("RGB")

    # Resize the image to 224x224 and prepare it for prediction
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# Streamlit UI
def show_waste_management():
    st.title("Smart Waste Management")
    st.subheader("Classify Waste and Identify the Correct Bin")

    # Instructions
    st.info(
        """
        - Upload an image of waste to classify.
        - The system will predict the type of waste and open the correct bin.
        - Supported formats: JPG, JPEG, PNG.
        """
    )

    # Create two columns
    col1, col2 = st.columns([2, 3])

    # Left column for image upload and classification
    with col1:
        st.subheader("Upload Waste Image")
        uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"])
        prediction_class = None  # Variable to store the prediction

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Perform classification
            with st.spinner("Classifying..."):
                prediction_class, confidence_score = classify_image(image)

            # Display results
            st.success(f"Prediction: {prediction_class}")
            st.info(f"Confidence Score: {confidence_score:.2f}")
        else:
            st.warning("Please upload an image to start the classification.")

    # Right column for displaying bin images
    with col2:
        st.subheader("Bins")
        cols = st.columns(2)

        # Loop through bin images and dynamically update based on classification
        for i, (bin_class, images) in enumerate(bin_images.items()):
            # st.info(f"{bin_class}")
            # st.info(f"{prediction_class}")
            # st.info(f"Bin Status: {'Open' if prediction_class == bin_class else 'Closed'}")
            bin_image = images["open"] if prediction_class == bin_class else images["closed"]
            with cols[i % 2]:
                st.image(bin_image, caption=f"{bin_class} Bin", use_container_width=True)
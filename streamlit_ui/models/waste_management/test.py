import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('hybrid_resnet50_effb1_classifier.h5')

# Define the image size expected by the model
IMG_SIZE = (224, 224)

# Define class names (assuming you have these from your training data)
class_names = ['biological', 'glass', 'paper', 'plastic'] # replace with your class labels

def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0 # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        return img_array
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Get user input for the image path
image_path = input("Enter the path to the image file: ")

# Preprocess the image
processed_image = preprocess_image(image_path)

if processed_image is not None:
    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_probability = predictions[0][predicted_class_index]

    # Get the class name from the index
    predicted_class_name = class_names[predicted_class_index]

    print(f"Predicted class: {predicted_class_name}")
    print(f"Confidence: {predicted_class_probability:.2f}")
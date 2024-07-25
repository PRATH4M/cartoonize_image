
import cv2
import numpy as np
import streamlit as st
from PIL import Image

def cartoonize_image(image, cartoon_level):
    # Convert to numpy array
    img = np.array(image)

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur to smooth the image
    gray = cv2.medianBlur(gray, 7)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Enhance edges using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Apply multiple passes of bilateral filtering to smooth the colors
    color = img
    for _ in range(5):
        color = cv2.bilateralFilter(color, 9, 75, 75)

    # Reduce the number of colors using k-means clustering
    Z = color.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    color = res.reshape((color.shape))

    # Combine edges with the color image
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges)

    # Adjust cartoonization level based on the slider value
    cartoon = cv2.addWeighted(img, 1 - cartoon_level, cartoon, cartoon_level, 0)

    return cartoon

# Streamlit UI
st.title("Cartoonize Your Image!")
st.write("Upload an image and see it transformed into a cartoon!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Cartoonizing..."):
        # Open the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        cartoon_level = st.slider("Cartoonization Level:", 0.0, 1.0, 0.5, step=0.05)

        cartoon = cartoonize_image(image, cartoon_level)

        st.image(cartoon, caption='Cartoonized Image.', use_column_width=True)

        # Get image dimensions
        width, height = image.size

        # Calculate maximum resolution based on image size
        max_resolution = min(width, height)

        # Resolution slider with dynamic range
        resolution = st.slider("Select resolution for the downloaded image:", 100, max_resolution, int(max_resolution * 0.8))

        # Resize the image based on the selected resolution directly in OpenCV format
        cartoon_resized = cv2.resize(cartoon, (resolution, resolution))

        # Directly convert the OpenCV image to bytes for download
        _, cartoon_bytes = cv2.imencode(".jpg", cartoon_resized)

        st.download_button(
            label="Download cartoonized image",
            data=cartoon_bytes.tobytes(),
            file_name="cartoonized_image.jpg",
            mime="image/jpg"
        )

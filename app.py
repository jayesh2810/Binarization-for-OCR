import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.title("Image Binarization for OCR")
st.write("Upload an image and select a binarization method to view the results.")

# 1.)
def otsu_binarization(img):
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

# 2.)
def adaptive_thresholding(img):
    binary_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    return binary_image

# 3.)
def niblack_binarization(img, window_size=15, k=-0.2):
    mean = cv2.boxFilter(img, cv2.CV_32F, (window_size, window_size))
    stddev = np.sqrt(cv2.boxFilter((img - mean)**2, cv2.CV_32F, (window_size, window_size)))
    threshold = mean + k * stddev
    binary_image = (img > threshold).astype(np.uint8) * 255
    return binary_image

# 4.)
def sauvola_binarization(img, window_size=15, k=0.5, r=128):
    mean = cv2.boxFilter(img, cv2.CV_32F, (window_size, window_size))
    stddev = np.sqrt(cv2.boxFilter((img - mean)**2, cv2.CV_32F, (window_size, window_size)))
    threshold = mean * (1 + k * ((stddev / r) - 1))
    binary_image = (img > threshold).astype(np.uint8) * 255
    return binary_image

# 5.)
def bernsen_binarization(img, window_size=15, contrast_threshold=15):
    min_img = cv2.erode(img, np.ones((window_size, window_size), np.uint8))
    max_img = cv2.dilate(img, np.ones((window_size, window_size), np.uint8))
    contrast = max_img - min_img
    mean = (max_img + min_img) / 2
    binary_image = np.where((contrast > contrast_threshold) & (img >= mean), 255, 0).astype(np.uint8)
    return binary_image

binarization_info = {
    "Otsu's Binarization": {
        "function": otsu_binarization,
        "points": [
            "Automatically finds the optimal global threshold to separate background and foreground.",
            "Ideal for images with bimodal histograms (two distinct peaks in intensity values).",
            "Threshold is chosen by minimizing the intra-class variance.",
        ],
        "formula": "Threshold = argmin(σ²_w + σ²_b)"
    },
    "Adaptive Thresholding": {
        "function": adaptive_thresholding,
        "points": [
            "Calculates threshold values locally based on small regions of the image.",
            "Useful for images with varying lighting or shadow effects.",
            "Two methods are commonly used: Mean and Gaussian adaptive thresholding.",
        ],
        "formula": "Threshold(x, y) = Mean(x, y) ± C (Gaussian for more weight)"
    },
    "Niblack's Binarization": {
        "function": niblack_binarization,
        "points": [
            "A local thresholding method that calculates the threshold using mean and standard deviation.",
            "Good for images with uneven illumination and textured backgrounds.",
            "Uses a parameter `k` to control the threshold sensitivity to local variance.",
        ],
        "formula": "Threshold(x, y) = Mean(x, y) + k * StdDev(x, y)"
    },
    "Sauvola's Binarization": {
        "function": sauvola_binarization,
        "points": [
            "An improvement over Niblack’s method, designed to handle noisy backgrounds.",
            "Uses local mean and standard deviation with a tuning parameter `k`.",
            "Suitable for documents with complex backgrounds and lighting conditions.",
        ],
        "formula": "Threshold(x, y) = Mean(x, y) * (1 + k * ((StdDev(x, y) / r) - 1))"
    },
    "Bernsen's Binarization": {
        "function": bernsen_binarization,
        "points": [
            "Divides the image into windows and uses the max and min pixel intensities within each window.",
            "Threshold is set to the average of max and min intensities if the contrast is high enough.",
            "Effective for low-contrast images with defined foreground and background regions.",
        ],
        "formula": "Threshold = (Max(x, y) + Min(x, y)) / 2 if contrast > T"
    },
}

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

method_name = st.selectbox("Choose a binarization method", list(binarization_info.keys()))

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    image_rgb = np.array(original_image)

    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    binarization_func = binarization_info[method_name]["function"]
    binarized_image = binarization_func(image_gray)

    st.write(f"### Selected Binarization Method: {method_name}")
    for point in binarization_info[method_name]["points"]:
        st.write(f"- {point}")
    st.write("#### Formula:")
    st.latex(binarization_info[method_name]["formula"])

    st.write("### Original Image vs. Binarized Image")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(image_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    axs[1].imshow(binarized_image, cmap="gray")
    axs[1].set_title(f"Binarized Image - {method_name}")
    axs[1].axis("off")
    
    st.pyplot(fig)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_histogram(ax, image, title):
    if len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    ax.plot(hist, color='orange')
    ax.set_title(title + " - Histogram", fontsize=10)
    ax.set_xlabel("Intensity", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)

def display_image_and_histogram(image, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title(title)
    ax1.axis("off")
    plot_histogram(ax2, image, title)
    plt.show()

def improve_image_quality(image_path, brightness_factor=1.2, contrast_factor=1.2, denoise_strength=10):
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read the image from {image_path}")
        return

    # Display original image
    display_image_and_histogram(img, "Original Image")

    # Apply brightness adjustment
    img_brightness = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    display_image_and_histogram(img_brightness, "Brightness Adjustment")

    # Apply contrast adjustment
    img_contrast = cv2.addWeighted(img_brightness, contrast_factor, np.zeros(img.shape, dtype=img.dtype), 0, 0)
    display_image_and_histogram(img_contrast, "Contrast Adjustment")

    # Apply Gaussian Blur
    img_blur = cv2.GaussianBlur(img_contrast, (5, 5), 0)
    display_image_and_histogram(img_blur, "Gaussian Blur")

    # Apply Bilateral Filter
    img_bilateral = cv2.bilateralFilter(img_blur, d=9, sigmaColor=75, sigmaSpace=75)
    display_image_and_histogram(img_bilateral, "Bilateral Filter")

    # Apply Gabor Filter
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    img_gabor = cv2.filter2D(img_bilateral, cv2.CV_8UC3, g_kernel)
    display_image_and_histogram(img_gabor, "Gabor Filter")

    # Apply Histogram Equalization
    img_gray = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2GRAY)
    img_equalized = cv2.equalizeHist(img_gray)
    display_image_and_histogram(cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2BGR), "Histogram Equalization")

    # Apply Laplacian Filter
    img_laplacian = cv2.Laplacian(img_equalized, cv2.CV_64F)
    img_laplacian = cv2.convertScaleAbs(img_laplacian)
    display_image_and_histogram(img_laplacian, "Laplacian Filter")

# Example usage
input_image_path = r'your_path'
improve_image_quality(input_image_path)

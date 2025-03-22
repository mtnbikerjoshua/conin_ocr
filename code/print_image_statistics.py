import cv2
import numpy as np
import sys
import math

def compute_entropy(image):
    """Compute the Shannon entropy of a grayscale image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()
    # Avoid log(0) by filtering out zero entries.
    entropy = -np.sum([p * math.log2(p) for p in hist_norm if p > 0])
    return entropy

def compute_gradient_magnitude(image):
    """Compute the average gradient magnitude using the Sobel operator."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return np.mean(magnitude)

def main(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    # Compute basic intensity statistics
    mean_intensity = np.mean(image)
    median_intensity = np.median(image)
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    intensity_range = max_intensity - min_intensity
    std_intensity = np.std(image)
    coeff_var = std_intensity / mean_intensity if mean_intensity != 0 else 0

    # Compute entropy of the image
    image_entropy = compute_entropy(image)

    # Compute Laplacian variance (a measure of image sharpness)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_variance = np.var(laplacian)

    # Compute average gradient magnitude (edge strength)
    avg_gradient = compute_gradient_magnitude(image)

    # Print the results
    print(f"Mean Intensity: {mean_intensity:.2f}")
    print(f"Median Intensity: {median_intensity:.2f}")
    print(f"Intensity Range: {min_intensity} - {max_intensity} (Difference: {intensity_range})")
    print(f"Coefficient of Variation: {coeff_var:.2f}")
    print(f"Entropy: {image_entropy:.2f}")
    print(f"Laplacian Variance: {laplacian_variance:.2f}")
    print(f"Average Gradient Magnitude: {avg_gradient:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    main(image_path)
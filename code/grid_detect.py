import sys
import numpy as np
from scipy.spatial import cKDTree
import cv2
import os
import re
import skimage.transform
import shutil
import warnings


def has_grid(image_file):
    # Load and threshold the image
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        warnings.warn(f"{image_file} is not readable")
        return None

    width = 1000
    scale = 1000/image.shape[1]
    height = round(scale * image.shape[0])

    image = cv2.resize(image, (width, height))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2)


    # Use morphological operations to find horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

    horizontal_lines = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, vertical_kernel)

    intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)

    # Get a list of (x, y) intersection coordinates
    def findCentroids(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
            else:
                centroids.append((0, 0))
        return centroids

    if(len(findCentroids(intersections)) < 200):
        return False
    else: 
        return True


image_dir = "data/Scanned Images"
image_files = [image_file for image_file in os.listdir(image_dir)]

for image_file in image_files:
    in_path = os.path.join(image_dir, image_file)
    out_path = os.path.join("data/Grid Images/", image_file)
    out_path_non = os.path.join("data/Non Grid Images/", image_file)
    if has_grid(in_path):
        shutil.copyfile(in_path, out_path)
    else:
        shutil.copyfile(in_path, out_path_non)
import numpy as np
import cv2


#### Line Detection ####
# --------------------------------------------------------------------- #

legajo = "2140T"

# Load and threshold the image
image = cv2.imread(f'data/Grid Images/Legajo_2140T 2024-07-29 12_33_00_page_3.png', cv2.IMREAD_GRAYSCALE)

width = 1000
scale = 1000/image.shape[1]
height = round(scale * image.shape[0])

image = cv2.resize(image, (width, height))
blurred = cv2.GaussianBlur(image, (5, 5), 0)
thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2)

# Use morphological operations to find horizontal and vertical lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 65))

horizontal_lines = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, horizontal_kernel)
vertical_lines = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, vertical_kernel)

boundaries = cv2.bitwise_xor(horizontal_lines, vertical_lines)

cv2.imwrite("output/boundaries.png", boundaries)
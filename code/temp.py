import numpy as np
import cv2 as cv

def show_wait_destroy(winname, img):
 cv.imshow(winname, img)
 cv.moveWindow(winname, 500, 0)
 cv.waitKey(0)
 cv.destroyWindow(winname)
 cv.waitKey(1)

img_path = '/Users/joshua/Desktop/Screenshot 2024-07-20 153351.jpeg'

# Read the image
img = cv.imread(img_path)

# Apply median blur to reduce noise
blurred = cv.medianBlur(img, 7)
blurred = cv.medianBlur(blurred, 7)
blurred = cv.medianBlur(blurred, 7)

# Dilate the image
dilated = cv.dilate(blurred, (5, 5), iterations=5)

# Convert the image to grayscale
gray = cv.cvtColor(dilated, cv.COLOR_BGR2GRAY)

# Show the grayscale image
show_wait_destroy('Grayscale Image', gray)

# Apply adaptive thresholding
thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

# Show the threshold image
show_wait_destroy('Threshold Image', thresh)

# Create a mask from the thresholded image
mask = cv.bitwise_not(thresh)

# Make the background of the original image transparent
result = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
result[:, :, 3] = mask

# Add transparent padding to the image
padded = cv.copyMakeBorder(result, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=(255, 255, 255, 0))

# Show the result
show_wait_destroy('Result', padded)

# Save the result
cv.imwrite('output/result.png', padded)
import numpy as np
import cv2 as cv

def show_wait_destroy(winname, img):
 cv.imshow(winname, img)
 cv.moveWindow(winname, 500, 0)
 cv.waitKey(0)
 cv.destroyWindow(winname)
 cv.waitKey(1)

img_path = '/Users/joshua/Desktop/result.jpeg'

# Read the image
img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

# Add transparent padding to the image
top = bottom = left = right = 50
padded_img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(255, 255, 255, 0))

# Save the padded image to desktop
output_path = '/Users/joshua/Desktop/padded_image.jpeg'
cv.imwrite(output_path, padded_img)

# Show the padded image
show_wait_destroy("Padded Image", padded_img)


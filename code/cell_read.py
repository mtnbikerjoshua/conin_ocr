import numpy as np
import cv2 as cv
import pytesseract
from cv2.ximgproc import thinning
from skimage.morphology import skeletonize, thin

def show_wait_destroy(winname, img):
 cv.imshow(winname, img)
 cv.moveWindow(winname, 500, 0)
 cv.waitKey(0)
 cv.destroyWindow(winname)
 cv.waitKey(1)

# Load the image
image = cv.imread('output/warped_124.png', cv.IMREAD_GRAYSCALE)

# Show the image
show_wait_destroy('image', image)

# Apply Gaussian blur
image = cv.GaussianBlur(image, (5, 5), 0)

# Apply canny edge detection
# image_can = cv.Canny(image, 50, 150)

# Show the image
# show_wait_destroy('Canny Edge Detection', image_can)

# Thresholding the image
(thresh, image_bin) = cv.threshold(image, 1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# Invert the image
image_bin = cv.bitwise_not(image_bin)

show_wait_destroy('image_bin', image_bin)

# Thinning the image
# thinned = skeletonize(image_bin, method='lee')

# Convert false/true to 0/255
# thinned = thinned.astype(np.uint8) * 255

# Show thinned image
# show_wait_destroy('Thinned', thinned)

# Get the contours
# contours, _ = cv.findContours(thinned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Convert the image to color
# image_col = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

# Get bounding rectangles
# bounding_rects = [cv.boundingRect(contour) for contour in contours]

# Draw bounding rectangles on the image
# for rect in bounding_rects:
#     x, y, w, h = rect
#     cv.rectangle(image_col, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the image with bounding rectangles
# show_wait_destroy('image with bounding rectangles', image_col)

# Remove noise
kernel = np.ones((3, 3), np.uint8)
image_bin = cv.morphologyEx(image_bin, cv.MORPH_OPEN, kernel, iterations=1)

# Sure background area
sure_bg = cv.dilate(image_bin, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=3)

# Sure foreground area
dist_transform = cv.distanceTransform(image_bin, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Show the foreground, background and unknown regions
show_wait_destroy('sure_fg', sure_fg)
show_wait_destroy('sure_bg', sure_bg)
show_wait_destroy('unknown', unknown)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# Show the markers
boundaries = cv.watershed(cv.cvtColor(image, cv.COLOR_GRAY2BGR), markers)

# Convert the image to color
image_col = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
# image_col[markers == -1] = [0, 0, 255]

# Show the image
# show_wait_destroy('Watershed Result', image_col)

# Get contours
contours, _ = cv.findContours(image_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
cv.drawContours(image_col, contours, -1, (0, 255, 0), 1)

# Show the image with contours
show_wait_destroy('image with contours', image_col)

# Calculate the convex hulls
convex_hulls = [cv.convexHull(contour, returnPoints=False) for contour in contours]

# Calculate the defects
defects = [cv.convexityDefects(contour, hull) for contour, hull in zip(contours, convex_hulls)]

# Get farthest points from the defects object
farthest_points = []
for i in range(0, len(contours)):
   if type(defects[i]) != type(None):
    for defect in defects[i]:
        far_idx, dist = defect[0, 2:4]
        far_point = contours[i][far_idx]
        if dist/256 > 8: farthest_points.append(far_point)


# Draw farthest points on the image
for point in farthest_points:
    cv.circle(image_col, tuple(point[0]), 2, (255, 0, 0), -1)

# Show the image with farthest points
show_wait_destroy('image with farthest points', image_col)


# Get complex contours
contours, _ = cv.findContours(image_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

def compute_slope(contour, point_index, k=5):
    """
    Computes the slope at a given point on the contour.
    
    :param contour: The contour points
    :param point_index: The index of the point on the contour
    :param k: The number of neighboring points to use for fitting a line
    :return: The slope (dy/dx) at the given point
    """
    # Ensure the point index is within the valid range
    if point_index < k or point_index >= len(contour) - k:
        raise ValueError("Point index is too close to the contour boundary.")

    # Extract the local neighborhood points
    neighborhood = contour[point_index - k:point_index + k + 1]
    
    # Fit a line to the neighborhood points
    [vx, vy, x, y] = cv.fitLine(neighborhood, cv.DIST_L2, 0, 0.01, 0.01)

    point = np.array(contour[point_index][0])
    prev_point = np.array(contour[point_index - 1][0])
    sign = np.sign(point - prev_point)
    
    vec = np.abs(np.array([vx[0], vy[0]])) * sign
    
    return vec

def show_destroy(winname, img):
 cv.imshow(winname, img)
 cv.moveWindow(winname, 500, 0)
 cv.waitKey(50)
 cv.destroyWindow(winname)
 cv.waitKey(1)

for point_index in range(5, len(contours[0]) - 5):
    vec = compute_slope(contours[0], point_index)

    # Get the coordinates of the point
    point = np.array(contours[0][point_index][0])

    # Draw the point and the slope direction on the image
    image_col2 = image_col.copy()
    cv.circle(image_col2, tuple(point), 1, (0, 0, 255), -1)
    cv.line(image_col2, tuple(point), tuple(np.round(point + 30*vec).astype(int)), (255, 0, 0), 1)

    show_destroy("Contour with slope", image_col2)
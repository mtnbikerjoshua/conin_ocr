import numpy as np
import cv2 as cv
from skimage import filters

def show_wait_destroy(winname, img):
 cv.imshow(winname, img)
 cv.moveWindow(winname, 500, 0)
 cv.waitKey(0)
 cv.destroyWindow(winname)
 cv.waitKey(1)

# Load and threshold the image
image = cv.imread('data/Legajo_2109_test.jpeg', cv.IMREAD_GRAYSCALE)
blurred = cv.GaussianBlur(image, (5, 5), 0)
_, thresholded = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)


# Use morphological operations to find horizontal and vertical lines
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (100, 1))
vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 200))

horizontal_lines = cv.morphologyEx(thresholded, cv.MORPH_OPEN, horizontal_kernel)
vertical_lines = cv.morphologyEx(thresholded, cv.MORPH_OPEN, vertical_kernel)

boundaries = cv.bitwise_xor(horizontal_lines, vertical_lines)

# Find the contour aproximation of the table
contours, _ = cv.findContours(boundaries, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
table_contour = sorted(contours, key=cv.contourArea, reverse=True)[:1][0]

epsilon = 0.1 * cv.arcLength(table_contour, True)
approx = cv.approxPolyDP(table_contour, epsilon, True)

# Extract corners of the table and sort in clockwise order starting from top-left
corners = approx.reshape(4, 2)

sorted_corners = np.zeros_like(corners)
sum_corners = corners.sum(axis=1)
sorted_corners[0] = corners[np.argmin(sum_corners)]
sorted_corners[2] = corners[np.argmax(sum_corners)]
diff_corners = np.diff(corners, axis=1)
sorted_corners[1] = corners[np.argmin(diff_corners)]
sorted_corners[3] = corners[np.argmax(diff_corners)]

# Transform the perspective of the image so that the table is a rectangle
src_points = np.array(sorted_corners, dtype=np.float32)
rect = cv.boundingRect(approx)
target_points = np.array([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]], [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]], dtype=np.float32)
pers_trans = cv.getPerspectiveTransform(src_points, target_points)
warped = cv.warpPerspective(image, pers_trans, (image.shape[1], image.shape[0]))

# Crop the image to extract the table
cropped = warped[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

# Repeat line detection on the cropped image
blurred = cv.GaussianBlur(cropped, (5, 5), 0)
_, thresholded = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (100, 1))
vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 200))

horizontal_lines = cv.morphologyEx(thresholded, cv.MORPH_OPEN, horizontal_kernel)
vertical_lines = cv.morphologyEx(thresholded, cv.MORPH_OPEN, vertical_kernel)

boundaries = cv.bitwise_xor(horizontal_lines, vertical_lines)

# Convert boundaries to 0-1 values
boundaries_bin = boundaries / 255

# Create horizontal and vertical kernels for convolution
# horizontal_kernel = np.vstack((np.full((1, 600), 0.5), np.ones((1, 600)), np.full((1, 600), 0.5)))
# vertical_kernel = np.hstack((np.full((600, 1), 0.5), np.ones((600, 1)), np.full((600, 1), 0.5)))

horizontal_convoluted = cv.filter2D(boundaries_bin, -1, np.ones((1, 600)))
vertical_convoluted = cv.filter2D(boundaries_bin, -1, np.ones((600, 1)))

horizontal_con_bin = cv.threshold(horizontal_convoluted, 300, 255, cv.THRESH_BINARY)[1]
vertical_con_bin = cv.threshold(vertical_convoluted, 300, 255, cv.THRESH_BINARY)[1]

con_bin = cv.bitwise_or(horizontal_con_bin, vertical_con_bin).astype(np.uint8)


# Detect lines using Probabilistic Hough Transform
lines = cv.HoughLinesP(boundaries, 1, np.pi/180, threshold=100, minLineLength=300, maxLineGap=50)

# Create a blank image to draw the detected lines
hough_lines = np.zeros_like(boundaries)

# Draw the detected lines on the blank image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(hough_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

# Combine the detected lines and the convoluted image
combined = cv.bitwise_or(hough_lines, con_bin)

# Set all border pixels to 255
combined[0, :] = 255
combined[-1, :] = 255
combined[:, 0] = 255
combined[:, -1] = 255

# Create a blank image to draw the contours
contour_image = np.zeros_like(combined)

# Find the contours of the inverted combined image
contours, _ = cv.findContours(combined, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# Find the convex hull of each contour
rotated_rects = []
bounding_rects = []
contour_corners = []
cropped_corners = []
for contour in contours:
    # Find the rotated bounding rectangle of the contour
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.intp(box)
    rotated_rects.append(box)
    
    # Find the bounding rectangle of the contour
    x, y, w, h = cv.boundingRect(contour)
    bounding_rects.append((x, y, w, h))
    
    # Extract the corners of the rotated rectangle
    corners = box.reshape(4, 2)
    contour_corners.append(corners)
    
    # Transform the corners to the coordinate space of the cropped image
    cropped_corners.append(corners - np.array([x, y]))

# Save the list of corners in the coordinate space of the cropped images
cropped_corners = np.array(cropped_corners)

# Draw the contours on the blank image
cv.drawContours(contour_image, rotated_rects, -1, (255, 255, 255), 2)

# Extend lines to the edge of the image
horizontal_lines = []
vertical_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    # Check if the line is vertical
    if x2 - x1 == 0:
        # Set the slope to infinity
        slope = float('inf')
        
        # Extend the line to the top and bottom edges of the image
        x1_extended = x1
        y1_extended = 0
        x2_extended = x2
        y2_extended = image.shape[0]
    else:
        # Calculate the slope of the line
        slope = (y2 - y1) / (x2 - x1)
    
        # Extend the line to the left and right edges of the image
        x1_extended = 0
        y1_extended = int(y1 - slope * x1)
        x2_extended = image.shape[1]
        y2_extended = int(y2 + slope * (x2_extended - x2))
    
    if slope > 5:
        vertical_lines.append([(x1_extended, y1_extended, x2_extended, y2_extended)])
    else:
        horizontal_lines.append([(x1_extended, y1_extended, x2_extended, y2_extended)])



# Cluster lines if either end of the line is close
def cluster_lines(lines, is_vertical):
    clusters = []
    for line in lines:
        if is_vertical:
            pos1 = 0
            pos2 = 2
        else:
            pos1 = 1
            pos2 = 3
        position = np.array([line[0][pos1], line[0][pos2]])
        found_cluster = False
        for cluster in clusters:
            cluster_pos = np.array([cluster[0][0][pos1], cluster[0][0][pos2]])
            if np.any(np.abs(position - cluster_pos) < 100):
                cluster.append(line)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([line])
    return clusters

# Find the median of each cluster
def median_cluster(cluster):
    lines = np.array(cluster)
    median = np.median(lines, axis=0)
    return [tuple(median[0])]

# Cluster lines if either end of the line is close
clustered_horizontal_lines = cluster_lines(horizontal_lines, is_vertical=False)
clustered_vertical_lines = cluster_lines(vertical_lines, is_vertical=True)

# Find the median of each cluster
median_horizontal_lines = [median_cluster(cluster) for cluster in clustered_horizontal_lines]
median_vertical_lines = [median_cluster(cluster) for cluster in clustered_vertical_lines]

# Draw the detected lines on the blank image
# for line in median_horizontal_lines + median_vertical_lines:
#     x1, y1, x2, y2 = line[0]
#     cv.line(cropped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Display the blank image with the detected lines
# show_wait_destroy('Average Lines', cropped)
# Use the bounding rects to crop into new images
cropped_images = []
for rect in bounding_rects:
    x, y, w, h = rect
    cropped_image = cropped[y:y+h, x:x+w]
    cropped_images.append(cropped_image)


    # Warp the cropped images using the cropped corners
    warped_images = []
    for image, corners in zip(cropped_images, cropped_corners):
        # Sort the corners in clockwise order starting from the top-left corner
        sorted_corners = np.zeros_like(corners)
        sum_corners = corners.sum(axis=1)
        sorted_corners[0] = corners[np.argmin(sum_corners)]
        sorted_corners[2] = corners[np.argmax(sum_corners)]
        diff_corners = np.diff(corners, axis=1)
        sorted_corners[1] = corners[np.argmin(diff_corners)]
        sorted_corners[3] = corners[np.argmax(diff_corners)]
        
        # Transform the perspective of the image so that the table is a rectangle
        src_points = np.array(sorted_corners, dtype=np.float32)
        rect = cv.boundingRect(src_points)
        target_points = np.array([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]], [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]], dtype=np.float32)
        pers_trans = cv.getPerspectiveTransform(src_points, target_points)
        
        # Warp the image
        warped_image = cv.warpPerspective(image, pers_trans, (image.shape[1], image.shape[0]))
        warped_images.append(warped_image)

# Save the warped images
for i, warped_image in enumerate(warped_images):
    cv.imwrite(f'output/warped_{i}.png', warped_image)
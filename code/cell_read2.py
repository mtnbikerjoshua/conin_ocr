import numpy as np
import cv2 as cv

def show_wait_destroy(winname, img):
 cv.imshow(winname, img)
 cv.moveWindow(winname, 500, 0)
 cv.waitKey(0)
 cv.destroyWindow(winname)
 cv.waitKey(1)

img_path = 'output/warped_118.png'

# Load the image
image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
show_wait_destroy('image', image)

# Color version of the image
image_col = cv.imread(img_path)

# Apply Gaussian blur to reduce noise
blurred = cv.GaussianBlur(image, (7, 7), 0)

# Apply canny edge detection
img_canny = cv.Canny(blurred, 50, 150)
show_wait_destroy('Canny Edge Detection', img_canny)

# Find contours
contours, _ = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# Create a list to store the bounding rectangles
bounding_rects = []

# Iterate over each contour
for contour in contours:
    # Get the bounding rectangle for the contour
    x, y, w, h = cv.boundingRect(contour)
    
    # Create a rectangle using the bounding rectangle coordinates
    rect = (x, y, w, h)
    
    # Add the rectangle to the list
    bounding_rects.append(rect)
    
# Draw the bounding rectangles on the image
for rect in bounding_rects:
    x, y, w, h = rect
    cv.rectangle(image_col, (x, y), (x + w, y + h), (0, 255, 0), 1)

# Show the image with bounding rectangles
show_wait_destroy('Bounding Rectangles', image_col)

# Combine overlapping and stacked rectangles
combined_rects = []

# Iterate over each rectangle
for rect in bounding_rects:
    x1, y1, w1, h1 = rect
    combined = False
    
    # Check if the current rectangle overlaps with any of the combined rectangles
    for combined_rect in combined_rects:
        x2, y2, w2, h2 = combined_rect
        
        # Calculate the intersection area
        intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        # Calculate the area of the smaller rectangle
        smaller_area = min(w1 * h1, w2 * h2)
        
        # Calculate the overlap ratio
        overlap_ratio = intersection_area / smaller_area

        # Calculate the x overlap
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) / min(w1, w2)
        
        # Check if the rectangles are stacked on top of each other
        stacked = (abs(y1 - (y2 + h2)) <= 10 or abs(y2 - (y1 + h1)) <= 10) and x_overlap > 0.5
        
        # If the overlap ratio is above a threshold or the rectangles are stacked, combine the rectangles
        if overlap_ratio > 0.5 or stacked:
            combined = True
            x1 = min(x1, x2)
            y1 = min(y1, y2)
            w1 = max(x1 + w1, x2 + w2) - x1
            h1 = max(y1 + h1, y2 + h2) - y1
            combined_rects.remove(combined_rect)
            break
    
    combined_rects.append((x1, y1, w1, h1))

# Draw the combined rectangles on the image
for rect in combined_rects:
    x, y, w, h = rect
    cv.rectangle(image_col, (x, y), (x + w, y + h), (0, 0, 255), 1)

# Show the image with combined rectangles
show_wait_destroy('Combined Rectangles', image_col)

# Select large rectangles that have the right aspect ratio and size to be a digit
selected_rects = []
size_threshold = 300

for rect in combined_rects:
    x, y, w, h = rect
    
    # Calculate the aspect ratio of the rectangle
    aspect_ratio = w / h
    
    # Check if the aspect ratio is within the threshold and the size is above the threshold
    if 0.2 <= aspect_ratio <= 1.2 and w * h >= size_threshold:
        selected_rects.append(rect)

# Draw the selected rectangles on the image
for rect in selected_rects:
    x, y, w, h = rect
    cv.rectangle(image_col, (x, y), (x + w, y + h), (255, 0, 0), 1)

# Show the image with selected rectangles
show_wait_destroy('Selected Rectangles', image_col)

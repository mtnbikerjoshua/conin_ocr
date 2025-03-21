import numpy as np
from scipy.spatial import cKDTree
import cv2
import skimage.transform

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)
    cv2.waitKey(1)

#### Line Detection ####
# --------------------------------------------------------------------- #

legajo = "2215"
legajo_n = 2215
legajo_encoded = legajo_n ^ 2344

# Load and threshold the image
image = cv2.imread(f'data/Data Tables/Legajo_{legajo}.jpeg', cv2.IMREAD_GRAYSCALE)

width = 1000
scale = 1000/image.shape[1]
height = round(scale * image.shape[0])

image = cv2.resize(image, (width, height))
blurred = cv2.GaussianBlur(image, (5, 5), 0)
thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2)
# _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

show_wait_destroy("Threshholded", thresholded)

# Use morphological operations to find horizontal and vertical lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 65))

horizontal_lines = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, horizontal_kernel)
vertical_lines = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, vertical_kernel)

boundaries = cv2.bitwise_xor(horizontal_lines, vertical_lines)
intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)

show_wait_destroy("Boundaries", boundaries)
show_wait_destroy("Intersections", intersections)


#### Align Image to Template ####
# --------------------------------------------------------------------- #

template_path = 'data/blank_template_1.png'
template_grayscale = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
_, template = cv2.threshold(template_grayscale, 0, 255, cv2.THRESH_BINARY)

# Resize the template to match the width of the input image
scale_factor = image.shape[1] / template.shape[1]
new_height = int(template.shape[0] * scale_factor)
new_width = image.shape[1]
template_resized = cv2.resize(template_grayscale, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def get_table_corners(image):
    # Find the contour aproximation of the table
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1][0]

    epsilon = 400
    approx = cv2.approxPolyDP(table_contour, epsilon, True)

    # Extract corners of the table and sort in clockwise order starting from top-left
    corners = approx.reshape(4, 2)

    sorted_corners = np.zeros_like(corners)
    sum_corners = corners.sum(axis=1)
    sorted_corners[0] = corners[np.argmin(sum_corners)]
    sorted_corners[2] = corners[np.argmax(sum_corners)]
    diff_corners = np.diff(corners, axis=1)
    sorted_corners[1] = corners[np.argmin(diff_corners)]
    sorted_corners[3] = corners[np.argmax(diff_corners)]
    return(sorted_corners)

detected_corners = get_table_corners(boundaries)
template_corners = get_table_corners(template_resized)


# Compute the homography matrix using the detected corners and template corners
H, status = cv2.findHomography(detected_corners, template_corners)

# Warp the boundaries image using the homography matrix
aligned_boundaries = cv2.warpPerspective(boundaries, H, (template_resized.shape[1], template_resized.shape[0]))
aligned_intersections = cv2.warpPerspective(intersections, H, (template_resized.shape[1], template_resized.shape[0]))
aligned_image = cv2.warpPerspective(image, H, (template_resized.shape[1], template_resized.shape[0]))

# Draw the resized template on the aligned image in red
aligned_display = cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2BGR)
aligned_display[:, :, 0][template_resized > 0] = 0
aligned_display[:, :, 1][template_resized > 0] = 0
aligned_display[:, :, 2][template_resized > 0] = 255


show_wait_destroy("Aligned Image with Template", aligned_display)


### Correct for Radial Distortion ###
# --------------------------------------------------------------------- #

template_horizontal = cv2.morphologyEx(template_resized, cv2.MORPH_OPEN, horizontal_kernel)
template_vertical = cv2.morphologyEx(template_resized, cv2.MORPH_OPEN, vertical_kernel)
template_intersections = cv2.bitwise_and(template_horizontal, template_vertical)

# Get a list of (x, y) corner coordinates
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

template_centroids = findCentroids(template_intersections)
detected_centroids = findCentroids(aligned_intersections)

# Display the template and detected centroids
black_background = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
for (x, y) in template_centroids:
    cv2.circle(black_background, (int(x), int(y)), 3, (255, 0, 0), -1)  # Blue
for (x, y) in detected_centroids:
    cv2.circle(black_background, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red
show_wait_destroy("Detected Centroids (Red) and Template Centroids (Blue)", black_background)

# Find the nearest neighbors for each detected centroid
tree = cKDTree(template_centroids)
distances, indices = tree.query(detected_centroids)

# Set a distance threshold to filter out outliers
threshold = 20
matched_pairs = {}
to_remove = set()

# Iterate through detected centroids and match with template centroids
for detected_idx, (dist, template_idx) in enumerate(zip(distances, indices)):
    if dist < threshold:
        if template_idx in matched_pairs:
            # If a template point already has a match, mark all as duplicates
            to_remove.add(template_idx)
            to_remove.add(matched_pairs[template_idx])
        matched_pairs[template_idx] = detected_idx

# Remove duplicated matches
template_points = []
detected_points = []

for template_idx, detected_idx in matched_pairs.items():
    if template_idx not in to_remove:
        template_points.append(template_centroids[template_idx])
        detected_points.append(detected_centroids[detected_idx])

# Display the template and detected centroids
black_background = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
for (x, y) in template_points:
    cv2.circle(black_background, (int(x), int(y)), 3, (255, 0, 0), -1)  # Blue
for (x, y) in detected_points:
    cv2.circle(black_background, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red
show_wait_destroy("Detected Centroids (Red) and Template Centroids (Blue)", black_background)


# Convert the control points into shape (N, 2) for skimage
template_points_sk = np.array(template_points, dtype=np.float32).reshape(-1, 2)
detected_points_sk = np.array(detected_points, dtype=np.float32).reshape(-1, 2)

tps = skimage.transform.ThinPlateSplineTransform()
tps.estimate(template_points_sk, detected_points_sk)
image_transformed = skimage.transform.warp(aligned_image, tps, order=0)

show_wait_destroy("Transformed Image", aligned_image)
show_wait_destroy("Transformed Image", image_transformed)


transformed_display = cv2.cvtColor(image_transformed, cv2.COLOR_GRAY2BGR)
transformed_display[:, :, 0][template_resized > 0] = 0
transformed_display[:, :, 1][template_resized > 0] = 0
transformed_display[:, :, 2][template_resized > 0] = 255


show_wait_destroy("Aligned Image with Template", transformed_display)

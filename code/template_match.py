import sys
import numpy as np
from scipy.spatial import cKDTree
import cv2
import os
import re
import skimage.transform
import math
import csv

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)
    cv2.waitKey(1)

# Maximum distance between template and image intersection to be considered a match
max_match_distance = 20

#### Line Detection ####
# --------------------------------------------------------------------- #

image_file = sys.argv[1]
legajo = re.sub(r'^.*Legajo_(.*) \d{4}-\d{2}-\d{2}.*$|^.*(Desconocido_.*) \d{4}-\d{2}-\d{2}.*$', r'\1\2', image_file)
page = re.sub(r'^.*page_(\d+).*$', r'\1', image_file)
# legajo_n = int(re.sub(r'\D', '', legajo))
# if 'T' in legajo:
#     legajo_n = legajo_n * 100 + 99
# elif 'G' in legajo:
#     legajo_n = legajo_n * 100 + 98
# legajo_encoded = legajo_n ^ 2344

print(f"Transforming Legajo {legajo} Page {page}")

# Load and threshold the image
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

width = 1000
scale = 1000/image.shape[1]
height = round(scale * image.shape[0])

image = cv2.resize(image, (width, height))

def compute_entropy(image):
    """Compute the Shannon entropy of a grayscale image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()
    # Avoid log(0) by filtering out zero entries.
    entropy = -np.sum([p * math.log2(p) for p in hist_norm if p > 0])
    return entropy
entropy = compute_entropy(image)
# C is assigned based on a sigmoid ranging from 2 to 37
C = -35 / (1 + math.exp(15 * (entropy - 3.2))) + 37
thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, C)
# _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# show_wait_destroy("Threshholded", thresholded)

# Use morphological operations to find horizontal and vertical lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

horizontal_lines = cv2.morphologyEx(thresholded, cv2.MORPH_ERODE, horizontal_kernel, iterations=1)
vertical_lines = cv2.morphologyEx(thresholded, cv2.MORPH_ERODE, vertical_kernel, iterations=1)

# show_wait_destroy("Horizonatal", horizontal_lines)

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 3))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 52))

horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_DILATE, horizontal_kernel, iterations=1)
vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_DILATE, vertical_kernel, iterations=1)

# show_wait_destroy("Horizonatal", horizontal_lines)



boundaries = cv2.bitwise_xor(horizontal_lines, vertical_lines)
intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)

# show_wait_destroy("Boundaries", boundaries)

# show_wait_destroy("Intersections", intersections)

cv2.imwrite("output/boundaries.png", boundaries)

#### Load the Templates ####
# --------------------------------------------------------------------- #

def load_template(template_name):
    template_path = f'data/Templates/{template_name}.png'
    template_grayscale = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    _, template = cv2.threshold(template_grayscale, 0, 255, cv2.THRESH_BINARY)

    # Resize the template to match the width of the input image
    scale_factor = image.shape[1] / template.shape[1]
    new_height = int(template.shape[0] * scale_factor)
    new_width = image.shape[1]
    template_resized = cv2.resize(template_grayscale, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return template_resized

template_names = [os.path.splitext(f)[0] for f in os.listdir('data/Templates') if f.endswith('.png')]
template_names.sort()
templates = [load_template(template_name) for template_name in template_names]


def get_table_corners(image):
    # Find the contour aproximation of the table
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1][0]
    table_hull = cv2.convexHull(table_contour)
    # Draw the table hull on the image
    hull_image = np.zeros_like(image)
    cv2.drawContours(hull_image, [table_hull], -1, (255, 255, 255), 2)
    # show_wait_destroy("Table Hull", hull_image)
    epsilon = 1
    approx = cv2.approxPolyDP(table_hull, epsilon, True)
    # Increase epsilon gradually until we get a quadrilateral
    while len(approx) != 4 and epsilon < 1000:
        epsilon += 1
        approx = cv2.approxPolyDP(table_hull, epsilon, True)

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
    return centroids

def calculate_points(template):
    detected_corners = get_table_corners(boundaries)
    template_corners = get_table_corners(template)

    # Compute the homography matrix using the detected corners and template corners
    H, status = cv2.findHomography(detected_corners, template_corners)

    # Warp the boundaries image using the homography matrix
    aligned_boundaries = cv2.warpPerspective(boundaries, H, (template.shape[1], template.shape[0]))
    aligned_intersections = cv2.warpPerspective(intersections, H, (template.shape[1], template.shape[0]))
    aligned_image = cv2.warpPerspective(image, H, (template.shape[1], template.shape[0]))
    aligned_horizontal_lines = cv2.warpPerspective(horizontal_lines, H, (template.shape[1], template.shape[0]))

    # Draw the resized template on the aligned image in red
    aligned_display = cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2BGR)
    aligned_display[:, :, 0][template > 0] = 0
    aligned_display[:, :, 1][template > 0] = 0
    aligned_display[:, :, 2][template > 0] = 255


    # show_wait_destroy("Aligned Image with Template", aligned_display)


    template_horizontal = cv2.morphologyEx(template, cv2.MORPH_OPEN, horizontal_kernel)
    template_vertical = cv2.morphologyEx(template, cv2.MORPH_OPEN, vertical_kernel)
    template_intersections = cv2.bitwise_and(template_horizontal, template_vertical)

    # show_wait_destroy("Template Intersections", template_intersections)

    template_centroids = findCentroids(template_intersections)
    detected_centroids = findCentroids(aligned_intersections)

    # Detect the left end of each horizontal line
    contours, _ = cv2.findContours(aligned_horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > 300]
    left_ends = []

    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        leftmost = tuple(map(int, cnt[cnt[:, :, 0].argmin()][0]))
        left_ends.append(leftmost)

    # Check if there are few centroids on the far left
    left_border_x = np.array(template_centroids).min(axis = 0)[0]
    far_left_centroids = [c for c in detected_centroids if c[0] < left_border_x + 20]
    if len(far_left_centroids) < 5:
        detected_centroids.extend(left_ends)

    # Display the template and detected centroids
    black_background = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for (x, y) in template_centroids:
        cv2.circle(black_background, (int(x), int(y)), 3, (255, 0, 0), -1)  # Blue
    for (x, y) in detected_centroids:
        cv2.circle(black_background, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red
    # show_wait_destroy("Detected Centroids (Red) and Template Centroids (Blue)", black_background)

    # Find the nearest neighbors for each detected centroid
    tree = cKDTree(template_centroids)
    distances, indices = tree.query(detected_centroids)

    # Set a distance threshold to filter out outliers
    matched_pairs = {}
    to_remove = set()

    # Iterate through detected centroids and match with template centroids
    for detected_idx, (dist, template_idx) in enumerate(zip(distances, indices)):
        if dist < max_match_distance:
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
    # show_wait_destroy("Detected Centroids (Red) and Template Centroids (Blue)", black_background)


    # Convert the control points into shape (N, 2) for skimage
    template_points = np.array(template_points, dtype=np.float32).reshape(-1, 2)
    detected_points = np.array(detected_points, dtype=np.float32).reshape(-1, 2)

    return([detected_points, template_points, aligned_image])

alignments = [calculate_points(template) for template in templates]
n_matches = [alignment[0].shape[0] for alignment in alignments]

best_alignment_index = np.argmax(n_matches)
detected_points, template_points, aligned_image = alignments[best_alignment_index]
template = templates[best_alignment_index]

tps = skimage.transform.ThinPlateSplineTransform()
tps.estimate(template_points, detected_points)
image_transformed = skimage.transform.warp(aligned_image, tps, order=0)

# show_wait_destroy("Transformed Image", aligned_image)
# show_wait_destroy("Transformed Image", image_transformed)


transformed_display = cv2.cvtColor(image_transformed, cv2.COLOR_GRAY2BGR)
transformed_display[:, :, 0][template > 0] = 0
transformed_display[:, :, 1][template > 0] = 0
transformed_display[:, :, 2][template > 0] = 255


# show_wait_destroy("Aligned Image with Template", transformed_display)

cv2.imwrite(f"output/template_matching/Legajo_{legajo}_page_{page}.png", transformed_display)
output_csv = "output/template_matching.csv"
file_exists = os.path.isfile(output_csv)

with open(output_csv, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['legajo', 'page', 'template_index'])
    writer.writerow([legajo, page, best_alignment_index])
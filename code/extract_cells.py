import sys
import numpy as np
import cv2
import re
import pandas as pd
import warnings
import os

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)
    cv2.waitKey(1)

image_file = sys.argv[1]
legajo = re.sub(r'^.*Legajo_(.*)_page.*$', r'\1', image_file)
page = re.sub(r'^.*page_(\d+).*$', r'\1', image_file)

legajo_n = int(re.sub(r'\D', '', legajo))
if 'T' in legajo:
    legajo_n = legajo_n * 100 + 99
elif 'G' in legajo:
    legajo_n = legajo_n * 100 + 98
legajo_encoded = legajo_n ^ 2344

image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
template_matches = pd.read_csv("output/template_matching.csv")

template_index = template_matches[(template_matches["legajo"] == legajo) & 
                                  (template_matches["page"] == int(page))]["template_index"].iloc[0]

def load_template(template_path):
    template_grayscale = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    _, template = cv2.threshold(template_grayscale, 0, 255, cv2.THRESH_BINARY)

    # Resize the template to match the width of the input image
    scale_factor = image.shape[1] / template.shape[1]
    new_height = int(template.shape[0] * scale_factor)
    new_width = image.shape[1]
    template_resized = cv2.resize(template_grayscale, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return template_resized

def has_left_line(image_file):
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
    
    template = load_template(f"data/Templates/blank_template_{template_index + 1}.png")
    template_contours, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(template_contours, key = cv2.contourArea)
    left_line_x, _, _, _ = cv2.boundingRect(largest_contour)

    left_line_centroids = [pt for pt in findCentroids(intersections) if pt[0] < left_line_x + 10 and pt[0] > left_line_x - 10]

    if(len(left_line_centroids) < 7):
        return False
    else: 
        return True
    
template = load_template(f"data/Templates/blank_template_{template_index + 1}.png")
if not has_left_line(image_file):
    template = load_template(f"data/Slicing Templates/slicing_template_{template_index + 1}.png")

skeleton = cv2.ximgproc.thinning(template)
contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

level2_contours = []
for i, cnt in enumerate(contours):
    # If the contour has a parent (hierarchy[0][i][3] != -1), it's a level 2 contour
    if hierarchy[0][i][3] != -1:
        level2_contours.append(cnt)

table_shape = ()
n_headers = 0
col_names = []
match template_index:
    case 0:
        table_shape = (18, 13)
        n_headers = 21
        col_names = ["date", "age", "weight", "height", "head_circumference", "bmi", "head_cir_z", "weight_age_z", "height_age_z", "weight_height_z", "bmi_z", "diagnosis", "signature"]
    case 1:
        table_shape = (19, 13)
        n_headers = 13
        col_names = ["date", "age", "weight", "height", "head_circumference", "bmi", "head_cir_z", "weight_age_z", "height_age_z", "weight_height_z", "bmi_z", "diagnosis", "signature"]
    case 2:
        table_shape = (19, 12)
        n_headers = 18
        col_names = ["date", "age", "weight", "height", "bmi", "head_cir_z", "weight_age_z", "height_age_z", "weight_height_z", "bmi_z", "diagnosis", "signature"]
    case 3:
        table_shape = (20, 14)
        n_headers = 22
        col_names = ["date", "age", "weight", "height", "head_circumference", "weight_age", "height_age", "weight_height", "bmi", "weight_z", "height_z", "weight_height_z", "diagnosis", "signature"]
    case 4:
        table_shape = (19, 12)
        n_headers = 12
        col_names = ["date", "age", "weight", "height", "bmi", "head_cir_z", "weight_age_z", "height_age_z", "weight_height_z", "bmi_z", "diagnosis", "signature"]

rects = [cv2.boundingRect(cnt) for cnt in level2_contours]
rects.reverse()
rects = np.array(rects[n_headers:]).reshape(table_shape + (4,))

cells = np.empty((rects.shape[0], rects.shape[1]), dtype=object)
for i in range(rects.shape[0]):
    for j in range(rects.shape[1]):
        x, y, w, h = rects[i, j]
        cells[i, j] = image[y:y+h, x:x+w]

# Save the images
output_path = f'output/chopped/{legajo}_page_{page}'
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i in range(cells.shape[0]):
    for j in range(cells.shape[1]):
        cv2.imwrite(f'{output_path}/cell_{legajo_encoded}_{col_names[j]}_{i}.png', cells[i, j])

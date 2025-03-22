import sys
import numpy as np
import cv2
import re
import pandas as pd

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)
    cv2.waitKey(1)

image_file = sys.argv[1]
legajo = re.sub(r'^.*Legajo_(.*)_page.*$', r'\1', image_file)
page = re.sub(r'^.*page_(\d+).*$', r'\1', image_file)

image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
template_matches = pd.read_csv("output/template_matching.csv")

template_index = template_matches[(template_matches["legajo"] == legajo) & 
                                  (template_matches["page"] == int(page))]["template_index"].iloc[0]

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

template = load_template(f"blank_template_{template_index + 1}")
skeleton = cv2.ximgproc.thinning(template)
contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

level2_contours = []
for i, cnt in enumerate(contours):
    # If the contour has a parent (hierarchy[0][i][3] != -1), it's a level 2 contour
    if hierarchy[0][i][3] != -1:
        level2_contours.append(cnt)

rects = [cv2.boundingRect(cnt) for cnt in level2_contours]

cropped_images = []
for rect in rects:
    x, y, w, h = rect
    cropped_image = image[y:y+h, x:x+w]
    cropped_images.append(cropped_image)

cv2.imwrite("output/test_image.png", cropped_images[-21])
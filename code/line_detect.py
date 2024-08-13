import numpy as np
import cv2 as cv
import base64
import requests
import os
import api_key

def show_wait_destroy(winname, img):
 cv.imshow(winname, img)
 cv.moveWindow(winname, 500, 0)
 cv.waitKey(0)
 cv.destroyWindow(winname)
 cv.waitKey(1)

#### Line Detection ####
# --------------------------------------------------------------------- #

legajo = 2109
legajo_encoded = legajo ^ 2344

# Load and threshold the image
image = cv.imread(f'data/Data Tables/Legajo_{legajo}.jpeg', cv.IMREAD_GRAYSCALE)
blurred = cv.GaussianBlur(image, (5, 5), 0)
_, thresholded = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)


# Use morphological operations to find horizontal and vertical lines
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (100, 1))
vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 200))

horizontal_lines = cv.morphologyEx(thresholded, cv.MORPH_OPEN, horizontal_kernel)
vertical_lines = cv.morphologyEx(thresholded, cv.MORPH_OPEN, vertical_kernel)

boundaries = cv.bitwise_xor(horizontal_lines, vertical_lines)

#### Crop Table ####
# --------------------------------------------------------------------- #

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


#### Line Detection on Cropped Image ####
# --------------------------------------------------------------------- #

# Repeat line detection on the cropped image
blurred = cv.GaussianBlur(cropped, (5, 5), 0)
_, thresholded = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (100, 1))
vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 200))

horizontal_lines = cv.morphologyEx(thresholded, cv.MORPH_OPEN, horizontal_kernel)
vertical_lines = cv.morphologyEx(thresholded, cv.MORPH_OPEN, vertical_kernel)

boundaries = cv.bitwise_xor(horizontal_lines, vertical_lines)

show_wait_destroy("boundaries", boundaries)


#### Convolution to Lengthen Lines ####
# --------------------------------------------------------------------- #

# Convert boundaries to 0-1 values
boundaries_bin = boundaries / 255

# Apply a convolution to lengthen lines
horizontal_convoluted = cv.filter2D(boundaries_bin, -1, np.ones((1, 500)))
vertical_convoluted = cv.filter2D(boundaries_bin, -1, np.ones((500, 1)))

horizontal_con_bin = cv.threshold(horizontal_convoluted, 240, 255, cv.THRESH_BINARY)[1]
vertical_con_bin = cv.threshold(vertical_convoluted, 240, 255, cv.THRESH_BINARY)[1]

con_bin = cv.bitwise_or(horizontal_con_bin, vertical_con_bin).astype(np.uint8)

show_wait_destroy("con_bin", con_bin)

# open_h = cv.morphologyEx(boundaries, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, 20)))
# open_v = cv.morphologyEx(boundaries, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (20, 1)))

bound_blur = cv.GaussianBlur(con_bin, (5, 7), 0)
bound_thresh = cv.threshold(bound_blur, 200, 255, cv.THRESH_BINARY)[1]
morph = cv.morphologyEx(bound_thresh, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (80, 7)))

show_wait_destroy("morph", morph)

### Probabilistic Hough Transform ###
# --------------------------------------------------------------------- #
lines = cv.HoughLinesP(boundaries, 1, np.pi/180, threshold=100, minLineLength=300, maxLineGap=50)

# Draw the detected lines on a blank image
hough_lines = np.zeros_like(boundaries)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(hough_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

# Combine the detected lines and the convoluted image
combined = cv.bitwise_or(hough_lines, con_bin)

### Split the table into cells ###
# --------------------------------------------------------------------- #

# Set all border pixels to 255
combined[0, :] = 255
combined[-1, :] = 255
combined[:, 0] = 255
combined[:, -1] = 255

show_wait_destroy("combined", combined)

# Find the cells in the table as contours
contours, _ = cv.findContours(combined, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# Sort the contours by area in descending order
contours = sorted(contours, key=cv.contourArea, reverse=True)[1:]

def sort_contours(contours, method="left-to-right"):
    # Initialize the reverse flag and the index of the bounding box coordinate
    reverse = False
    i = 0
    
    # Handle whether we need to sort in reverse or not
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    
    # Handle sorting by rows first (y-coordinates) or columns (x-coordinates)
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    
    # Extract the bounding boxes
    bounding_boxes = [cv.boundingRect(c) for c in contours]
    
    # Sort the contours and bounding boxes based on the i-th coordinate
    sorted_contours = sorted(zip(contours, bounding_boxes),
                             key=lambda b: b[1][i], reverse=reverse)
    
    return zip(*sorted_contours)

def group_contours_by_rows(contours, threshold=10):
    # Sort contours by y-coordinate
    contours, bounding_boxes = sort_contours(contours, method="top-to-bottom")
    
    rows = []
    current_row = []
    
    for i, (contour, bbox) in enumerate(zip(contours, bounding_boxes)):
        if i == 0:
            current_row.append(contour)
            previous_bbox = bbox
            continue
        
        # Check if the current contour is close enough to the previous one to be considered in the same row
        if abs(bbox[1] - previous_bbox[1]) < threshold:
            current_row.append(contour)
        else:
            # Start a new row
            rows.append(current_row)
            current_row = [contour]
        
        previous_bbox = bbox
    
    # Append the last row
    if current_row:
        rows.append(current_row)
    
    return rows

def sort_table_cells(contours):
    # Group contours by rows
    rows = group_contours_by_rows(contours)
    
    sorted_contours = []
    for row in rows:
        # Sort each row by x-coordinate (left-to-right)
        sorted_row, _ = sort_contours(row, method="left-to-right")
        sorted_contours.extend(sorted_row)
    
    return sorted_contours


sorted_contours = sort_table_cells(contours)


# Find the bounding rects of the cells in preparation for perspective transformation and cropping
rotated_rects = []
bounding_rects = []
contour_corners = []
cropped_corners = []
for contour in sorted_contours:
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

cropped_corners = np.array(cropped_corners)

# Convert the cropped image to color
cropped_color = cv.cvtColor(cropped, cv.COLOR_GRAY2BGR)

# Use the bounding rects to crop into new images
cropped_images = []
for rect in bounding_rects:
    x, y, w, h = rect
    cropped_image = cropped[y:y+h, x:x+w]
    if cropped_image.shape[0] > 30 and cropped_image.shape[1] > 30 and cropped_image.size > 8000: cropped_images.append(cropped_image)


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

cells = np.reshape(np.array(cropped_images[8:], dtype=object), (19, 13))

# Save the images
output_path = f'output/{legajo}'
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i in range(cells.shape[0]):
    for j in range(cells.shape[1]):
        match j:
            case 0:
                column = "date"
            case 1:
                column = "age"
            case 2:
                column = "weight"
            case 3:
                column = "height"
            case 4:
                column = "head_circumference"
            case 5:
                column = "bmi"
            case 6:
                column = "head_cir_z"
            case 7:
                column = "weight_age_z"
            case 8:
                column = "height_age_z"
            case 9:
                column = "weight_height_z"
            case 10:
                column = "bmi_z"
            case 11:
                column = "diagnosis"
            case 12:
                column = "signature"
        cv.imwrite(f'{output_path}/cell_{legajo_encoded}_{column}_{i}.png', cells[i, j])


# Encode each cell as base64
encoded_cells = np.empty(cells.shape, dtype=object)
for i in range(cells.shape[0]):
    for j in range(cells.shape[1]):
        _, encoded_cell = cv.imencode('.png', cells[i, j])
        encoded_cell_base64 = base64.b64encode(encoded_cell).decode('utf-8')
        encoded_cells[i, j] = encoded_cell_base64

def transcribe_cell(cell, column):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key.api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "These images are cells from a table of nutritional data for children under 5 years old. All responses should be reasonable measurements for a child of that age. Each image will be labeled with the column it comes from. The date column contains the date of data collection in d/m/y format, and should be after 2015. The age column contains the child’s age in years and months with “a” meaning “año” or year, and “m” meaning “mes” or month. The weight column contains the child’s weight in kg. The height column contains the child’s height in cm. For any image that does not contain a number, return “empty”. Any number may contain a decimal point or comma and may be followed by its units. A comma means the same as a decimal point."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"{column}"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{cell}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()["choices"][0]["message"]["content"]

transcriptions = np.empty(cells.shape, dtype = object)
for i in range(1, 4):
    for j in range(4):
        cell = encoded_cells[i, j]
        match j:
            case 0:
                column = "date"
            case 1:
                column = "age"
            case 2:
                column = "weight"
            case 3:
                column = "height"
            case _:
                column = "other"
        # transcriptions[i, j] = transcribe_cell(cell, column)

print(transcriptions)
        
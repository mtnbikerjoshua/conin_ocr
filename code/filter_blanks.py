import shutil
import cv2
import os

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)
    cv2.waitKey(1)

# Get all .png file paths in 'data/chopped' directory (recursive)
image_paths = [
    os.path.join(root, f)
    for root, _, files in os.walk('data/chopped')
    for f in files
    if f.lower().endswith('.png')
]

def is_blank_image(image_path, area_threshold=50):
    """
    Returns True if the total area of contours in the image is less than area_threshold.
    """
    image = cv2.imread(image_path)
    if image is None:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    h, w = thresh.shape
    thresh = thresh[5:h-5, 5:w-5]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)
    # show_wait_destroy(str(total_area), thresh)
    return total_area < area_threshold

for image_file in image_paths:
    is_blank_image(image_file)

for image_file in image_paths:
    filename = os.path.basename(image_file)
    out_path = os.path.join("output/To Transcribe/", filename)
    out_path_non = os.path.join("output/Blank Images/", filename)
    if is_blank_image(image_file):
        shutil.copyfile(image_file, out_path_non)
    else:
        shutil.copyfile(image_file, out_path)
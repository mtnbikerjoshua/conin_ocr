import os
from pdf2image import convert_from_path

# Define the directories
input_dir = 'data/Scanned Documents'
output_dir = 'data/Scanned Images'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(input_dir, filename)
        images = convert_from_path(pdf_path)
        
        # Save each page as an image
        for i, image in enumerate(images):
            image_filename = f"{os.path.splitext(filename)[0]}_page_{i + 1}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path, 'PNG')

print("PDF to image conversion completed.")
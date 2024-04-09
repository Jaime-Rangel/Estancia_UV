import os
from PIL import Image

def convert_to_rgb(image_path):
    try:
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
            img.save(image_path)
            print(f"Converted {image_path} to RGB format.")
        else:
            print(f"{image_path} is already in RGB format.")
    except Exception as e:
        print(f"Error converting {image_path}: {str(e)}")

def convert_images_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                convert_to_rgb(image_path)

# Specify the root directory containing images
root_directory = './train'

# Convert images in the root directory and its subdirectories
convert_images_in_directory(root_directory)

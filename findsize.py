import os
from PIL import Image

def is_image(file_path):
    try:
        with Image.open(file_path):
            return True
    except:
        return False

def check_image_shape(directory, target_shape):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if is_image(file_path):
            with Image.open(file_path) as img:
                if img.size != target_shape:		    
                
                    print(f"{filename} does not have the desired shape.")

# Set your target image shape (width, height)

print(os.getcwd())
localpath = os.getcwd();

target_shape = (604, 452)
# Set your directory path
directory_path = localpath

check_image_shape(directory_path, target_shape)

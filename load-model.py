from keras.models import Model, load_model
import matplotlib.pyplot as plt
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import keras
import numpy as np
import os

input_dir = "/Users/jaime/Macbook IA Dropbox/jaime rangel/Universidad Veracruzana/Materias/Estancia/Full_dataset/input/test"
img_size = (256, 256)

trained_model = keras.saving.load_model("./betamodel_v2.keras", custom_objects=None, compile=True, safe_mode=False)

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
val_input_img_paths = input_img_paths[:]
X_test = np.zeros((len(val_input_img_paths), img_size[0], img_size[1], 3), dtype=np.uint8)

def load_img_masks(test_img_path):
 
    for i, test_id in enumerate(test_img_path):    
        test_img = tf_io.read_file(test_id)
        test_img = tf_io.decode_png(test_img, channels=3)
        test_img = tf_image.resize(test_img, img_size)
        test_img = tf_image.convert_image_dtype(test_img, "float32")

        X_test[i] = test_img


    return X_test

def display_mask(i):
    """Quick utility to display a model's binary prediction mask."""
    preds_test_thresh = (val_preds >= 0.5).astype(np.uint8)
    test_img = preds_test_thresh[i, :, :, 0]

    # Adds a subplot at the 3rd position 
    fig.add_subplot(rows, columns, 2) 
    
    # showing image 
    plt.imshow(test_img,cmap='gray') 
    plt.axis('off') 
    plt.title("Third")

X_test = load_img_masks(val_input_img_paths)

# setting values to rows and column variables 
rows = 1
columns = 2

# Display results for validation image #0
i = 13

# create figure 
fig = plt.figure(figsize=(10, 7))

# Adds a subplot at the 1st position 
fig.add_subplot(rows, columns, 1) 

# showing image 
plt.imshow(X_test[i, :, :, 0]) 
plt.axis('off') 
plt.title("Original")

val_preds = trained_model.predict(X_test)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.
plt.show()


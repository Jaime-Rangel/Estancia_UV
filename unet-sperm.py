import os
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from keras.utils import load_img
from PIL import Image as pilimage
from PIL import ImageOps as pilimageops
import keras
from keras import layers
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import tensorflow as tf
from keras.layers import Input,Dropout, Lambda, Conv2D,Conv2DTranspose,MaxPooling2D,concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.io import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
import random
import tensorflow as tf

from numba import cuda
cuda.select_device(0)
cuda.close()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess =tf.compat.v1.Session(config=config)

input_dir = "./spiders/input"
target_dir = "./spiders/target"

#Hyperparameters
img_size = (512, 512)

epochs = 200
batch_size = 7
val_samples = 17
epoch_patience = 15
model_learning_rate = 0.0004
validation_split = 0.2

input_ext = ".jpg"
target_ext = ".png"

"""
## Set aside a validation split
"""

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(input_ext)
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(target_ext)
    ]
)

# Define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range=0.2,  # set range for random shear
    zoom_range=0.2,  # set range for random zoom
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False,  # randomly flip images vertically
    fill_mode='nearest'  # fill mode for handling newly created pixels
)

print("Number of samples:", len(input_img_paths))

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)

train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Split our img paths into a training and a validation set
X_train = np.zeros((len(train_input_img_paths), img_size[0], img_size[1], 3), dtype=np.uint8)
Y_train = np.zeros((len(train_input_img_paths), img_size[0], img_size[1], 2), dtype=np.bool_)
X_test = np.zeros((len(val_input_img_paths), img_size[0], img_size[1], 3), dtype=np.uint8)
Y_test = np.zeros((len(val_target_img_paths), img_size[0], img_size[1], 2), dtype=np.bool_)

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)

# # Display input image #7
# disp1 = np.asarray(pilimage.open(input_img_paths[9]))
# plt.imshow(disp1)

# # Display auto-contrast version of corresponding target (per-pixel categories)
# img = pilimageops.autocontrast(load_img(target_img_paths[9]))

# plt.imshow(img)

def get_dataset(
     batch_size,
     img_size,
     input_img_paths,
     target_img_paths,
     max_dataset_len=None,
 ):
    """Returns a TF Dataset."""

def load_img_masks(input_img_path, target_img_path, val_input_img_path,val_target_img_path):
    
    for i, image_id in enumerate(input_img_path):            
        
        input_img = tf_io.read_file(image_id)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")
        
        X_train[i] = input_img

    for i, mask_id in enumerate(target_img_path):            
        target_img = tf_io.read_file(mask_id)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        Y_train[i] = target_img

    for i, test_id in enumerate(val_input_img_path):    
        test_img = tf_io.read_file(test_id)
        test_img = tf_io.decode_png(test_img, channels=3)
        test_img = tf_image.resize(test_img, img_size)
        test_img = tf_image.convert_image_dtype(test_img, "float32")

        X_test[i] = test_img

    for i, mask_id in enumerate(val_target_img_path):            
        target_img = tf_io.read_file(mask_id)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        Y_test[i] = target_img

    return X_train, Y_train, X_test, Y_test

"""
## Prepare U-Net Xception-style model
"""

def get_model():

    inputs = Input((img_size[0], img_size[1], 3))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(2, (1, 1), activation='sigmoid') (c9)

    return inputs,outputs

[X_train, Y_train, X_test, Y_test] = load_img_masks(train_input_img_paths,train_target_img_paths,val_input_img_paths,val_target_img_paths)

train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size)

mmodel = get_model()
model = Model(inputs=[mmodel[0]], outputs=[mmodel[1]])

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=model_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

"""
## Train the model
"""

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
filepath = "model.h5"

earlystopper = EarlyStopping(patience=epoch_patience, monitor='val_loss', verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')

callbacks_list = [earlystopper, checkpoint]

history = model.fit(X_train,Y_train,validation_split = validation_split, epochs=epochs, callbacks=callbacks_list)

"""
## Visualize predictions
"""

# Generate predictions for all images in the validation set

valid_dataset = get_dataset(
    batch_size, img_size, val_input_img_paths, val_target_img_paths
)

def display_mask(i):
    """Quick utility to display a model's binary prediction mask."""
    preds_test_thresh = (val_preds >= 0.5).astype(np.uint8)
    test_img = preds_test_thresh[i, :, :, 0]

    # Adds a subplot at the 3rd position 
    fig.add_subplot(rows, columns, 3) 
    
    # showing image 
    plt.imshow(test_img,cmap='gray') 
    plt.axis('off') 
    plt.title("Predicted")
    print()

# setting values to rows and column variables 
rows = 3
columns = 2

# Display results for validation image #0

i = 0

# Display input image
#display(Image(filename=val_input_img_paths[i]))

# create figure 
fig = plt.figure(figsize=(10, 7)) 

# disp2 = np.asarray(pilimage.open(val_input_img_paths[i]))
# plt.imshow(disp2)

# Adds a subplot at the 1st position 
fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(X_test[i, :, :, 0]) 
plt.axis('off') 
plt.title("Original") 

# Display ground-truth target mask
#img = ImageOps.autocontrast(load_img(val_target_img_paths[i]))
#img = pilimageops.autocontrast(load_img(val_target_img_paths[i]))

# Adds a subplot at the 1st position 
fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(Y_test[i, :, :, 0],cmap='gray')
plt.axis('off') 
plt.title("Mask") 

val_preds = model.predict(X_test)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.

model.save('./spiders_v2.keras')  # The file needs to end with the .keras extension

fig.add_subplot(rows, columns, 4) 
plt.plot(history.history['accuracy'], label='train') 
plt.plot(history.history['val_accuracy'], label='test') 
plt.legend() 

fig.add_subplot(rows, columns, 5) 
plt.plot(history.history['loss'], label='train loss') 
plt.plot(history.history['val_loss'], label='validation loss') 

plt.legend() 
plt.show()
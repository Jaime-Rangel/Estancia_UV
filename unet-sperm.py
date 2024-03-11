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
from skimage.transform import resize
import random

input_dir = "/Users/jaime/Desktop/sperm_datasetv2/input_renames"
target_dir = "/Users/jaime/Desktop/sperm_datasetv2/target_renames"
img_size = (160, 160)
num_classes = 1
batch_size = 16

val_samples = 900

"""
## Set aside a validation split
"""

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpeg")
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png")
    ]
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

Y_train = np.zeros((len(train_input_img_paths), img_size[0], img_size[1], 1), dtype=np.bool_)

X_test = np.zeros((len(val_input_img_paths), img_size[0], img_size[1], 3), dtype=np.uint8)

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

def load_img_masks(input_img_path, target_img_path, test_img_path):
    for i, image_id in enumerate(input_img_path):            
        
        input_img = tf_io.read_file(image_id)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")
        
        # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
        
        # insert the image into X_train
        X_train[i] = input_img

    for i, mask_id in enumerate(target_img_path):            
        target_img = tf_io.read_file(mask_id)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        Y_train[i] = target_img

    
    for i, test_id in enumerate(test_img_path):    
        test_img = tf_io.read_file(test_id)
        test_img = tf_io.decode_png(test_img, channels=3)
        test_img = tf_image.resize(test_img, img_size)
        test_img = tf_image.convert_image_dtype(test_img, "float32")

        X_test[i] = test_img


    return X_train, Y_train, X_test


    # For faster debugging, limit the size of data
    # if max_dataset_len:
    #     input_img_paths = input_img_paths[:max_dataset_len]
    #     target_img_paths = target_img_paths[:max_dataset_len]

    # dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    # dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)

    # #Just run
    # [im1,im2] = load_img_masks(input_img_paths[0],target_img_paths[0])
    # disp1 = np.asarray(im2)
    # plt.imshow(disp1,cmap="gray", interpolation='nearest')

    # return dataset.batch(batch_size)

"""
## Prepare U-Net Xception-style model
"""

def get_mode_v2():
    inputs = Input((img_size[0], img_size[1], 3))

    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    return inputs,outputs

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    
    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Build model
#model = get_model(img_size, num_classes)


[X_train, Y_train, X_test] = load_img_masks(train_input_img_paths,train_target_img_paths,val_input_img_paths)

# Instantiate dataset for each split
# Limit input files in `max_dataset_len` for faster epoch training time.
# Remove the `max_dataset_len` arg when running with full dataset.


mmodel = get_mode_v2()
model = Model(inputs=[mmodel[0]], outputs=[mmodel[1]])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

"""
## Train the model
"""

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
# model.compile(
#    optimizer=keras.optimizers.legacy.Adam(1e-4),
#     loss="binary_crossentropy",  # Change the loss function
#     metrics=["accuracy"],
# )

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 100

filepath = "model.h5"

earlystopper = EarlyStopping(patience=20, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')

callbacks_list = [earlystopper, checkpoint]

# model.fit(
#     train_dataset,
#     epochs=epochs,
#     validation_data=valid_dataset,
#     callbacks=callbacks_list,
#     verbose=1,
# )

# X_train = np.asarray(X_train).astype('float32').reshape((-1,1))
# Y_train = np.asarray(Y_train).astype('uint8').reshape((-1,1))

history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, 
                    callbacks=callbacks_list)

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

    #mask_pred = (val_preds[i] > 0.5).astype(np.uint8)  # Convertir a binario con un umbral de 0.5

    # mask = np.expand_dims(mask_pred, axis=-1)
    # mask = np.squeeze(mask, axis=(2, 3))
    #mask = np.asarray(array_redimensionado)
    #img = pilimageops.autocontrast(keras.utils.array_to_img(mask))

    # Adds a subplot at the 3rd position 
    fig.add_subplot(rows, columns, 3) 
    
    # showing image 
    plt.imshow(test_img,cmap='gray') 
    plt.axis('off') 
    plt.title("Third")

# setting values to rows and column variables 
rows = 2
columns = 2

# Display results for validation image #0

i = 100

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
img = pilimageops.autocontrast(load_img(val_target_img_paths[i]))

# Adds a subplot at the 1st position 
fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(img) 
plt.axis('off') 
plt.title("Mask") 

val_preds = model.predict(X_test)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.

model.save('./betamodel_v1.keras')  # The file needs to end with the .keras extension

print()
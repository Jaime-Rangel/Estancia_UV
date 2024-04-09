import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 10

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# sample 25 mnist digits from train dataset
indexes = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]


# # plot the 25 mnist digits
# plt.figure(figsize=(5,5))
# for i in range(len(indexes)):
#     plt.subplot(5, 5, i + 1)
#     image = images[i]
#     plt.imshow(image, cmap='gray')
#     nonzeroind = np.nonzero(labels[i])

#     plt.title("Digit: {}".format(nonzeroind[0]))
#     plt.axis('off')

num_index = 0
val_preds = model.predict(x_test)
labels = np.argmax(val_preds, axis=1)

# pred_img = x_test[0]
# label_pred = val_preds[0]
# nonzeroind = np.nonzero(label_pred[0])

# plt.imshow(pred_img, cmap='gray')

plt.title(' Prediction: ' + str(labels[num_index]))
plt.imshow(y_test[num_index], cmap='gray')
plt.grid(False)
plt.axis('off')

plt.show()

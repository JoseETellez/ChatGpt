# Ask questions to ChatGpt
```
# Generic Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,re
import random
from PIL import Image

# TensorFlow / Keras Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, Conv1D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

# Plotting Libraries
import matplotlib.pyplot as plt

# SKLearn Libraries
from sklearn.model_selection import train_test_split

#ROC curve
from sklearn.metrics import roc_curve, roc_auc_score

from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive/Dataset

train_images = "/content/drive/MyDrive/Dataset/train"
test_images = "/content/drive/MyDrive/Dataset/test"

image_size = (180, 180)
batch_size = 10

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_images,
    #validation_split=0,
    #subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

print(train_ds)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_images,
    #validation_split=0,
    #subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

data = []
labels = []

for images, batch_labels in train_ds:
    data.append(images)
    labels.append(batch_labels)

data = tf.concat(data, axis=0)
labels = tf.concat(labels, axis=0)

normalized_data = data / 255.0

labels.shape, normalized_data.shape

model = Sequential()

model.add(Conv2D(32, kernel_size = (1, 1), activation='relu', input_shape=(180,180,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation = 'softmax'))

# Compile the network
x = inputs = Input(shape=(180, 180,3))
x = Conv1D(
    filters=50,
    kernel_size=4,
    strides=2,
    activation="relu",
)(x)
x = Conv1D(filters=50, kernel_size=4,strides=1, activation="relu")(x)
#x = GlobalAveragePooling1D()(x)
x = Flatten()(x)
x = Dense(50, activation="relu")(x)
x = Dense(10, activation="relu")(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)

epochs = 25

callbacks = [
    keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=5,
    verbose=0,
    mode="min")
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    normalized_data, labels,
    epochs=epochs,
    callbacks=callbacks
)

plt.figure(figsize=(4,3), dpi=120)
plt.plot(history.history['loss'], label = 'Train')

img = keras.utils.load_img("/content/drive/MyDrive/Dataset/test/Null/roti_0011_jpg.rf.33598e0e863c45a4e6907367ed496da2.jpg", target_size=image_size)

predictions = []
true_labels = []
for images, labels in test_ds:
    batch_predictions = model.predict(images).squeeze(1)
    predictions.extend(batch_predictions)
    true_labels.extend(labels.numpy())

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(true_labels, predictions)
auc_score = roc_auc_score(true_labels, predictions)

plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

img_array = keras.utils.img_to_array(img)
plt.imshow(img_array.astype("uint8"))

img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array) #probability to predict class 1
score = float(predictions[0])

train_ds.class_names

print(f"This image is {100 * (1 - score):.2f}% {train_ds.class_names[0]} \
and {100 * score:.2f}% {train_ds.class_names[1]}")

score



from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import time
import psutil
import os
import cv2
import scipy
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense
from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# SEQUENTIAL API model

# i want to see how many resurses this program consumes & the time it takes
process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss / 1024 / 1024  # MB
start_time = time.time()

# loading and splitting data
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = datasets.mnist.load_data()

# normalizeing data
x_train = x_train_orig.astype('float32') / 255.0
x_test = x_test_orig.astype('float32') / 255.0

# reshape
y_train = y_train_orig.T
y_test = y_test_orig.T

nr_batch_train, _, _ = x_train.shape
nr_batch_test, _, _ = x_test.shape

# applying the filters to the train array
for i in range(nr_batch_train):
    x_train_img = x_train[i]

    # converting to PIL images to apply the grayscale filter
    x_train_img = Image.fromarray((x_train_img * 255).astype(np.uint8))

    # apply the grayscale filer
    x_train_gray = ImageOps.grayscale(x_train_img)

    # convert back to numpy array
    x_train_np = np.array(x_train_gray).astype('float32') / 255.0

    # back to the dataset
    x_train[i] = x_train_np

# applying the filters to the test array
for i in range(nr_batch_test):
    x_test_img = x_test[i]

    # converting to PIL images to apply the grayscale filter
    x_test_img = Image.fromarray((x_test_img * 255).astype(np.uint8))

    # apply the grayscale filer
    x_test_gray = ImageOps.grayscale(x_test_img)

    # convert back to numpy array
    x_test_np = np.array(x_test_gray).astype('float32') / 255.0

    # back to the dataset
    x_test[i] = x_test_np

# one hot encoding
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

# build the model
model = tf.keras.Sequential([
    ZeroPadding2D(padding=(3,3), data_format='channels_last', input_shape=(28, 28, 1)),
    Conv2D(strides=1, filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(strides=1, filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(strides=(2,2)),

    Conv2D(strides=1, filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(strides=1, filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(strides=(2,2)),

    Flatten(),
    tfl.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_start = time.time()
history = model.fit(x_train, y_train, epochs=10, batch_size=16)
train_time = time.time() - train_start

eval_start = time.time()
loss, accuracy = model.evaluate(x_test, y_test)
eval_time = time.time() - eval_start

total_time = train_time + eval_time
process = psutil.Process(os.getpid())
end_memory = process.memory_info().rss / 1024 / 1024  # MB
memory_used = end_memory - start_memory

print("The test loss is: ", loss)
print("The test accuracy is: ", accuracy)
print("Total time ", total_time, " seconds")
print("Memory used ", memory_used, " MB")

# confussion matrix
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)   # model chosen class
y_true = np.argmax(y_test.numpy(), axis=1)   # corect class

# build the confussion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# display the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix - MNIST")
plt.show()

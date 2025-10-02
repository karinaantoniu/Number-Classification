import os
import cv2
import time
import scipy
import keras
import psutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl
from tensorflow.keras import datasets
from models.imagenet import mobilenetv2
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# i want to see how many resurses this program consumes & the time it takes
process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss / 1024 / 1024  # MB
start_time = time.time()

# loading and splitting data
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = datasets.mnist.load_data()

# normalizeing data
x_train = x_train_orig.astype('float32') / 255.0
x_test = x_test_orig.astype('float32') / 255.0

def preproccess(images):
	images = images.reshape(-1, 28, 28, 1)

	image_res = []
	for img in images:
		img_res = tf.image.resize(img, [32, 32])
		image_res.append(img_res)
	image_res = np.array(image_res)

	img_rgb = tf.repeat(image_res, 3, axis=-1)

	return img_rgb

x_train = preproccess(x_train)
x_test = preproccess(x_test)

print(x_train.shape)

# one hot encoding
y_train = tf.one_hot(y_train_orig.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test_orig.astype(np.int32), depth=10)

# build the model
model = keras.applications.MobileNetV2(
    input_shape=(32, 32, 3), alpha=0.5, weights=None, include_top=True, classes=10
)

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
print("Total time ", total_time, "seconds")
print("Memory used ", memory_used, "MB")

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
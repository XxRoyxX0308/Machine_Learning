import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers


train = tfds.load('mnist', split='train', as_supervised=True)
test = tfds.load('mnist', split='test', as_supervised=True)


def format_image(image, label):
  image = tf.cast(image, dtype=tf.float32)  # 資料型態轉換 float32
  image = image / 255.0                     # 正規劃數值到 [0, 1] 區間
  return  image, label

BATCH_SIZE = 32
BUFFER_SIZE = 10000

train_batches = train.cache().shuffle(BUFFER_SIZE).map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

model = tf.keras.Sequential([
  layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
  layers.MaxPool2D(pool_size=(2, 2)),
  layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
  layers.MaxPool2D(pool_size=(2, 2)),
  layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
  layers.MaxPool2D(pool_size=(2, 2)),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(10, activation='softmax')
])

print(model.summary())

model.compile(
  optimizer='adam', 
  loss='sparse_categorical_crossentropy', 
  metrics=['accuracy']
)

EPOCH = 3
history = model.fit(
  train_batches, 
  validation_data=test_batches, 
  epochs=EPOCH
)

model.save('./mnist_save.h5')

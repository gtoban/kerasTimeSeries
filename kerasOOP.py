import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten

print(tf.version.VERSION)
print(tf.keras.__version__)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=10, activation="relu", input_shape=(3000,1))) #shape batch, steps, channels
model.add(Conv1D(filters=16, kernel_size=10, activation="relu"))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])


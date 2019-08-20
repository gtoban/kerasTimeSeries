import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

print(tf.version.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential([
layers.Dense(64, activation="relu", input_shape=(3000,)),
layers.Dense(64, activation="relu"),
layers.Dense(10, activation="softmax")])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

f = open("input002.csv")
line = f.readline().strip()
data = np.zeros(3000,)
labels = np.zeros(1,)
sample = 0
while (line):
    arow = line.split(",")
    labels[sample] = 1 if arow[0] ==
    
    line = f.readline().strip()
                                                  

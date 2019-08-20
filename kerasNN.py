import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

print(tf.version.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential([
layers.Dense(3000, activation="relu", input_shape=(3000,)),
layers.Dense(3000, activation="relu"),
layers.Dense(3000, activation="relu"),
layers.Dense(3000, activation="relu"),
layers.Dense(3000, activation="relu"),
layers.Dense(2, activation="softmax")])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

record_count = 0
with open("input002.csv") as f:
    for line in f:
        if (line.strip()):
            record_count += 1
f = open("input002.csv")
line = f.readline().strip()
data = np.zeros((record_count,3000))
labels = np.zeros((record_count,2))
sample = 0
while (line):
    arow = line.split(",")
    labels[sample][1 if arow[0] == 'W' else 0] = 1
    measure_count = 0
    for ame in arow[1:]:
        data[sample][measure_count] = ame
    
    sample += 1
    line = f.readline().strip()
    
model.fit(data, labels, epochs=30, batch_size=record_count)


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

print("shape:", data.shape)
data = np.expand_dims(data,axis=2)
print("shape:", data.shape)
model.fit(data, labels, epochs=1, batch_size=record_count)


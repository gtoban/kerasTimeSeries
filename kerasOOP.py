import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten


class ann_data(object):
    def __init__(self):
        self.dataPath = ""
        self.outputPath = ""

    def readData(self):
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

        data = np.expand_dims(data,axis=2)
        #print("shape:", data.shape)
        return data, labels, record_count

class keras_ann(object):
    def __init__(self):
        self.dataPath = ""
        self.outputPath = ""

    def convModel(self, no_filters=[32,16]):
        model = Sequential()
        model.add(Conv1D(filters=no_filters[0], kernel_size=10, activation="relu", input_shape=(3000,1))) #shape batch, steps, channels
        model.add(Conv1D(filters=no_filters[1], kernel_size=10, activation="relu"))
        model.add(Flatten())
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer='adam', #tf.train.AdamOptimizer(0.001),
                    loss='categorical_crossentropy', #tf.keras.losses.categorical_crossentropy,
                    metrics=['acc']) #tf.keras.metrics.categorical_accuracy
        return model


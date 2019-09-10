import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix

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
            labels[sample][1 if arow[0] == 'R' else 0] = 1
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
        self.inputShape = (3000,1)
        self.metric_Obj = ann_data()

    def f1_score(self, y_true, y_pred):
        #y_true = get_value(y_true_tf)
        #y_pred = get_value(y_pred_tf)
        #true_neg = 0
        #true_pos = 0
        #false_neg = 0
        #flase_pos = 0
        #for i in range(y_true.shape[0]):
        #    true = np.argmax(y_true[i])
        #    pred = np.argmax(y_pred[i])
        #    if (pred == 1):
        #        if (true == pred):
        #            true_pos += 1
        #        else:
        #            false_pos += 1
        #    else:
        #        if (true == pred):
        #            true_neg += 1
        #        else:
        #            false_neg += 1
        #precision = true_pos/(true_pos+false_pos)
        #recall = true_pos/(true_pos+false_neg)        
        #f1_score =  2 * ((precision*recall)/(precision+recall))
        ##f1_score, update_op = tf.contrib.metrics.f1_score(y_true_tf,y_pred_tf)
        #return to_dense(f1_score)

        #=============
        # SOURCE:
        # https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
        #=============
        
        true_pos = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
        poss_pos = K.sum(K.round(K.clip(y_true,0,1)))
        recall = true_pos / (poss_pos + K.epsilon())
        pred_pos = K.sum(K.round(K.clip(y_true,0,1)))
        precision = true_pos/(pred_pos + K.epsilon())
        f1_score =  2 * ((precision*recall)/(precision+recall+K.epsilon()))
        
    def convModel(self, modelArgs=[
        {
            'layer': 'conv1d',
            'no_filters': 64,
            'kernal_size':10,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 32,
            'kernal_size':10,
            'activation':'relu'
        },{
            'layer': 'flatten'
        },{
            'layer': 'dense',
            'output': 2,
            'activation':'softmax'
        },{
            'layer': 'compile',
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics':['acc']
        }]):
        model = Sequential()

        #
        # For First Layer, input requried
        #
        modelArg= modelArgs[0]
        if (modelArg['layer'] == 'conv1d'):
            model.add(Conv1D(filters=modelArg['no_filters'], kernel_size=modelArg['kernal_size'], activation=modelArg['activation'], input_shape=self.inputShape)) #shape batch, steps, channels
        elif (modelArg['layer'] == 'dense'):
                model.add(Dense(activation=modelArg['activation'], input_shape=self.inputShape))
            
        for modelArg in modelArgs[1:]:
            if (modelArg['layer'] == 'conv1d'):
                model.add(Conv1D(filters=modelArg['no_filters'], kernel_size=modelArg['kernal_size'], activation=modelArg['activation'])) #shape batch, steps, channels
            elif (modelArg['layer'] == 'flatten'):
                model.add(Flatten())

            elif (modelArg['layer'] == 'dense'):
                model.add(Dense(modelArg['output'], activation=modelArg['activation']))
            elif (modelArg['layer'] == 'maxpool1d'):
                model.add(MaxPooling1D(pool_size=modelArg['pool_size'],
                                           strides=modelArg['strides'],
                                           padding=modelArg['padding']))
            elif (modelArg['layer'] == 'avgpool1d'):
                model.add(AveragePooling1D(pool_size=modelArg['pool_size'],
                                           strides=modelArg['strides'],
                                           padding=modelArg['padding']))
            elif (modelArg['layer'] == 'compile'):
                for i in range(len(modelArg['metrics'])):
                    if (modelArg['metrics'][i] == 'f1score'):
                        modelArg['metrics'][i] = self.f1_score
                        break
                model.compile(optimizer=modelArg['optimizer'], #tf.train.AdamOptimizer(0.001),
                    loss=modelArg['loss'], #tf.keras.losses.categorical_crossentropy,
                    metrics=[metrics.CategoricalAccuracy(), metrics.TrueNegatives(), metrics.TruePositives()]) #tf.keras.metrics.categorical_accuracy
        #model.add(Conv1D(filters=no_filters[0], kernel_size=10, activation="relu", input_shape=self.inputShape)) #shape batch, steps, channels
        #model.add(Conv1D(filters=no_filters[1], kernel_size=10, activation="relu"))
        #model.add(Flatten())
        #model.add(Dense(2, activation="softmax"))
        #model.compile(optimizer='adam', #tf.train.AdamOptimizer(0.001),
        #            loss='categorical_crossentropy', #tf.keras.losses.categorical_crossentropy,
        #            metrics=['acc']) #tf.keras.metrics.categorical_accuracy
        return model

    def parameterSearch(self, paramSets, X, Y, numSplits):
        # create CV dat LOOV 
        #numSplits = 2
        Kf = KFold(n_splits=numSplits)
        #for each parameter set
        # make a model
        #
        #X = [0,1,2,3,4,5,6,7,8,9]
        modelFile = open("fileModel.csv", 'w')
        resultFile = open("fileResult.csv",'w')
        resultFile.write("modelNum|True Neg|False Neg|True Pos|False Pos|F1_score|Acc\n")
        modelNum = 0
        for paramSet in paramSets:
            #print(paramSet)
            
            modelFile.write(str(modelNum) + "|")
            json.dump(paramSet, modelFile)
            modelFile.write("\n")
            
            
            try:
                model = self.convModel(paramSet)            
                j = 0
                for trainInd, testInd in Kf.split(X):
                    model.fit(X[trainInd], Y[trainInd], batch_size=None, verbose=0)
                    Ypred = np.zeros((testInd.shape[0],Y.shape[1]))
                    Yi = 0
                    for pred in np.argmax(model.predict(X[testInd]), axis=1):
                        Ypred[Yi][pred] = 1
                        Yi += 1

                    #print(Y[testInd], Ypred)
                    confusionMatrix = multilabel_confusion_matrix(Y[testInd], Ypred)
                    #confusionMatrix = confusion_matrix(np.argmax(Y[testInd], axis=1), np.argmax(Ypred, axis=1))
                    #print(confusionMatrix)
                    #print('f1_score:',f1_score(Y[testInd], Ypred, average='macro'))
                    resultFile.write(str(modelNum) + "|")
                    for row in confusionMatrix:
                        for el in row:
                            resultFile.write(str(el) + "|")

                    resultFile.write(str(f1_score(Y[testInd], Ypred, average='macro')) + "|\n")
                
                    j+=1
            
            except (Exception):
                resultFile.write("error\n")
            modelNum+=1
        modelFile.close()
        resultFile.close()

        

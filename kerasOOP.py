import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix
import signal
import sys


class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True
    
class ann_data(object):
    def __init__(self):
        self.dataPath = ""
        self.outputPath = ""

    def readData(self, fnames=["input002.csv","input142.csv"]):
        record_count = 0
        for fname in fnames:
            with open(self.dataPath + fname) as f:
                for line in f: 
                    if (line.strip()):
                        record_count += 1
        sample = 0
        for fname in fnames:
            data = np.zeros((record_count,3000))
            labels = np.zeros((record_count,2))

            RVNR = [0,0]
            f = open(self.dataPath + fname)
            line = f.readline().strip()
            while(line):
                arow = line.split(",")
                labels[sample][0 if arow[0] == 'R' else 1] = 1
                RVNR[0 if arow[0] == 'R' else 1] += 1
                measure_count = 0
                for ame in arow[1:]:
                    data[sample][measure_count] = ame
    
                sample += 1
                line = f.readline().strip()
            f.close()
            print(f"{fname} -> REM: {RVNR[0]}; NonREM: {RVNR[1]}")
            
                    
        data = np.expand_dims(data,axis=2)
        
        #print("shape:", data.shape)
        return data, labels, record_count
    
class keras_ann(object):
    def __init__(self):
        self.dataPath = ""
        self.outputPath = ""
        self.inputShape = (3000,1)
        self.killer = GracefulKiller()

    def updatePaths(self, dataPath = '', outputPath = ''):
        self.outputPath = outputPath
        self.dataPath = dataPath
        
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
            
        #
        # For all other layers
        #
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
                #NOTE: metrics are not used for training and therefor not really needed. The loss is the important one
                model.compile(optimizer=modelArg['optimizer'], #tf.train.AdamOptimizer(0.001),
                    loss=modelArg['loss']) #tf.keras.losses.categorical_crossentropy,
                    #metrics=['acc']) #metrics.CategoricalAccuracy(), metrics.TrueNegatives(), metrics.TruePositives()]) #tf.keras.metrics.categorical_accuracy
                
        return model

    def parameterSearch(self, paramSets, X, Y, numSplits=2,valData=None, epochs=1, batchSize=None):
        # create CV dat LOOV 
        #numSplits = 2
        
        Kf = StratifiedKFold(n_splits=numSplits)
        #for each parameter set
        # make a model
        #
        #X = [0,1,2,3,4,5,6,7,8,9]
        modelFile = open(self.outputPath + "fileModel.csv", 'w')
        resultFile = open(self.outputPath + "fileResult.csv",'w')
        resultFile.write("modelNum|True REM|False REM|False NonREM|True NonREM|Acc|Sens|Spec|Recall|Precision|f1score\n")
        modelNum = 0
        for paramSet in paramSets:
            #print(paramSet)
            modelFile.write(str(modelNum) + "|")
            json.dump(paramSet, modelFile)
            modelFile.write("\n")
            print("\n\n=================\nTesting Model " + str(modelNum) + "\n=================\n", flush=True)
            
            try:
                model = self.convModel(paramSet)            
                j = 0
                for trainInd, testInd in Kf.split(X, np.argmax(Y,axis=1)):
                    
                    model.fit(X[trainInd], Y[trainInd], batch_size=batchSize, verbose=0, validation_data=valData, epochs=epochs)
                    Ypred = np.zeros((testInd.shape[0],Y.shape[1]))
                    Yi = 0
                    for pred in np.argmax(model.predict(X[testInd], batch_size=None), axis=1):
                        Ypred[Yi][pred] = 1
                        Yi += 1

                    
                    #NOTE:
                    #confusionMatrix = multilabel_confusion_matrix(Y[testInd], Ypred)[0]
                    ##print(confusionMatrix)
                    ##confusionMatrix = confusion_matrix(np.argmax(Y[testInd], axis=1), np.argmax(Ypred, axis=1))
                    ##print(confusionMatrix)
                    ##print('f1_score:',f1_score(Y[testInd], Ypred, average='macro'))
                    #resultFile.write(str(modelNum) + "|")
                    ##for row in confusionMatrix:
                    ##    for el in row:
                    ##        resultFile.write(str(el) + "|")
                    ##"modelNum|True REM|False NonREM|False REM|True NonREM|Acc|Sens|Spec|Recall|Precision|f1score\n"
                    #
                    #tn = confusionMatrix[0][0]
                    #fn = confusionMatrix[1][0]
                    #tp = confusionMatrix[1][1]
                    #fp = confusionMatrix[0][1]

                    tp=tn=fn=fp=0
                    Yi= 0
                    for y in Y[testInd]:
                        tp += Ypred[Yi][0]*y[0]
                        fp += max(Ypred[Yi][0]-y[0],0)
                        tn += Ypred[Yi][1]*y[1]
                        fn += max(Ypred[Yi][1]-y[1],0)
                        Yi+=1
                       
                    acc=sens=spec=prec=rec=f1=0
                    acc=(tp+tn)/(tp+tn+fp+fn)
                    if (tp+fn > 0):
                        sens=tp/(tp+fn)
                    if (tn+fp > 0):
                        spec=tn/(tn+fp)
                    if (tp+fp > 0):
                        prec=tp/(tp+fp)
                    if (tp+fn > 0):
                        rec=tp/(tp+fn)
                    if (prec+rec > 0):
                        f1=2*((prec*rec)/(prec+rec))
                    resultFile.write(f"{modelNum}|{tp:.3f}|{fp:.3f}|{fn:.3f}|{tn:.3f}|{acc:.3f}|{sens:.3f}|{spec:.3f}|{rec:.3f}|{prec:.3f}|{f1:.3f}\n")
                    #resultFile.write(str(f1_score(Y[testInd], Ypred, average='macro')) + "|\n")
                
                    j+=1
            
            except Exception as e:
                resultFile.write("error\n")
                print(str(e))
                
            modelNum+=1
            
            if self.killer.kill_now:
                resultFile.write("killed\n")
                print("killed")
                break
            
        modelFile.close()
        resultFile.close()

        

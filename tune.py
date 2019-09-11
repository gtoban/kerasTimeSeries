import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from kerasOOP import keras_ann, ann_data
import os

def main():
    myAnn = keras_ann()
    myData = ann_data()

    data,labels,recordCount = myData.readData()
    overfitData, overfitLabels, overfitRecordCount = myData.readData(fname="input142.csv")
    data = data[:100] 
    labels = labels[:100] 
    
    modelArgs = getModels()
    addToModels(modelArgs)
    myAnn.updatePaths(outputPath = os.path.dirname(os.path.realpath(__file__)) + "/")
    myAnn.parameterSearch(modelArgs,data,labels,numSplits=2,valData=(overfitData,overfitLabels), epochs=1, batchSize=int(recordCount/10)+1)
    

def addToModels(modelArgs):
    #low freq to high freq
    modelArgs.append(
        [
        {
            'layer': 'conv1d',
            'no_filters': 66,
            'kernal_size':5,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 20,
            'kernal_size':10,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 10,
            'kernal_size':15,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 5,
            'kernal_size':20,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 3,
            'kernal_size':25,
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
        }])
    #high to low freq
    modelArgs.append(
        [
        {
            'layer': 'conv1d',
            'no_filters': 3,
            'kernal_size':25,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 5,
            'kernal_size':20,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 10,
            'kernal_size':15,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 20,
            'kernal_size':10,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 66,
            'kernal_size':5,
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
        }])

    layers5 = [{
            'layer': 'conv1d',
            'no_filters': 3,
            'kernal_size':25,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 5,
            'kernal_size':20,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 10,
            'kernal_size':15,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 20,
            'kernal_size':10,
            'activation':'relu'
        },{
            'layer': 'conv1d',
            'no_filters': 66,
            'kernal_size':5,
            'activation':'relu'
        }]
    # A small model for each of the 5 conv layers with and without maxpooling. Potential preparation for ensemble.
    for layer in layers5:
        #without max pooling
        modelArgs.append([])
        modelArgs[-1].append(layer)
        modelArgs[-1].append({
            'layer': 'flatten'
        })
        modelArgs[-1].append({
            'layer': 'dense',
            'output': 2,
            'activation':'softmax'
        })
        modelArgs[-1].append({
            'layer': 'compile',
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics':['acc']
        })

        #with max pooling
        modelArgs.append([])
        modelArgs[-1].append(layer)
        modelArgs[-1].append({
            'layer': 'maxpool1d',
            'pool_size': 10,
            'strides':None,
            'padding':'valid'
        })
        modelArgs[-1].append({
            'layer': 'flatten'
        })
        modelArgs[-1].append({
            'layer': 'dense',
            'output': 2,
            'activation':'softmax'
        })
        modelArgs[-1].append({
            'layer': 'compile',
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics':['acc']
        })
    
    #Mix and match features randomly (5 convolution layers (order: 5->120, 4->120, 3->60, 2->20, total->320), maxPooling (5) (1600), number of dense layers (1-5) etc (nodes: 50,100,200,400,500,800))
    possibleNumOfNodes = [50,100,200,400,500,800]
    possiblePoolSize = [5,10,20,40,80]
    for i in range(100):
        modelArgs.append([])
        #how many conv layers
        numConLayers = np.random.randint(2,6)

        #add a random list of layers of specified length
        for layerIndex in np.random.randint(0,5,size=numConLayers):
            modelArgs[-1].append(layers5[layerIndex])
            # add maxpooling?
            if (np.random.randint(2) == 1):
                modelArgs[-1].append({
                    'layer': 'maxpool1d',
                    'pool_size': possiblePoolSize[np.random.randint(len(possiblePoolSize))],
                    'strides':None,
                    'padding':'valid'})
        # add a flatten layer
        modelArgs[-1].append({
            'layer': 'flatten'
        })
        
        # pick a random number of dense layers primarily picking 1
        numDenseLayers = min(max(int(np.random.normal(1,scale=2.5)), 1),5)
        for layerNum in range(numDenseLayers-1):
            modelArgs[-1].append({
                'layer': 'dense',
                'output': possibleNumOfNodes[np.random.randint(len(possibleNumOfNodes))],
                'activation':'softmax'})
        modelArgs[-1].append({
            'layer': 'dense',
            'output': 2,
            'activation':'softmax'
        })
        modelArgs[-1].append({
            'layer': 'compile',
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics':['acc']
        })
    
def getModels():
    modelArgs = []
    #modelArgs.append(
    #    [
    #    {
    #        'layer': 'conv1d',
    #        'no_filters': 64,
    #        'kernal_size':10,
    #        'activation':'relu'
    #    },{
    #        'layer': 'conv1d',
    #        'no_filters': 32,
    #        'kernal_size':10,
    #        'activation':'relu'
    #    },{
    #        'layer': 'flatten'
    #    },{
    #        'layer': 'dense',
    #        'output': 2,
    #        'activation':'softmax'
    #    },{
    #        'layer': 'compile',
    #        'optimizer': 'adam',
    #        'loss': 'categorical_crossentropy',
    #        'metrics':['acc']
    #    }])
    #
    #modelArgs.append(
    #    [
    #    {
    #        'layer': 'conv1d',
    #        'no_filters': 128,
    #        'kernal_size':10,
    #        'activation':'relu'
    #    },{
    #        'layer': 'conv1d',
    #        'no_filters': 64,
    #        'kernal_size':10,
    #        'activation':'relu'
    #    },{
    #        'layer': 'flatten'
    #    },{
    #        'layer': 'dense',
    #        'output': 2,
    #        'activation':'softmax'
    #    },{
    #        'layer': 'compile',
    #        'optimizer': 'adam',
    #        'loss': 'categorical_crossentropy',
    #        'metrics':['acc']
    #    }])

    #Automatic Sleep Stage Scoring with Single-Channel EEG Using CNNs
    modelArgs.append(
        [
        {
            'layer': 'conv1d',
            'no_filters': 20,
            'kernal_size':200,
            'activation':'relu'
        },{
            'layer': 'maxpool1d',
            'pool_size': 20,
            'strides':None,
            'padding':'valid'
        },{
            'layer': 'conv1d',
            'no_filters': 400,
            'kernal_size':20,
            'activation':'relu'
        },{
            'layer': 'maxpool1d',
            'pool_size': 10,
            'strides':None,
            'padding':'valid'
        },{
            'layer': 'flatten'
        },{
            'layer': 'dense',
            'output': 500,
            'activation':'relu'
        },{
            'layer': 'dense',
            'output': 500,
            'activation':'relu'
        },{
            'layer': 'dense',
            'output': 2,
            'activation':'softmax'
        },{
            'layer': 'compile',
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics':['acc']
        }])

    #Real-time human activity recognition from accelerometer data using CNN
    modelArgs.append(
        [
        {
            'layer': 'conv1d',
            'no_filters': 196,
            'kernal_size':12,
            'activation':'relu'
        },{
            'layer': 'maxpool1d',
            'pool_size': 4,
            'strides':None,
            'padding':'valid'
        },{
            'layer': 'flatten'
        },{
            'layer': 'dense',
            'output': 500,
            'activation':'relu'
        },{
            'layer': 'dense',
            'output': 2,
            'activation':'softmax'
        },{
            'layer': 'compile',
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics':['acc']
        }])


    return modelArgs

def f1_score(self, y_true, y_pred):
    true_neg = 0
    true_pos = 0
    false_neg = 0
    flase_pos = 0
    for i in len(y_true):
        true = np.argmax(y_true[i])
        pred = np.argmax(y_pred[i])
        if (pred == 1):
            if (true == pred):
                true_pos += 1
            else:
                false_pos += 1
        else:
            if (true == pred):
                true_neg += 1
            else:
                false_neg += 1
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    return 2 * ((precision*recall)/(precision+recall))


main()

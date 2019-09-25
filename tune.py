import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from kerasOOP import keras_ann, ann_data
import os


#NOTE: FLAG FOR TESTING SMALL MODELS ONLY!!
smallModelOnly = True
numOfInputFiles = 8
def main():
    myAnn = keras_ann()
    myData = ann_data(dataPath="/nfshome/gst2d/eegData/")
    
    
    modelArgs = [] #getModels() small models only for now!
    #addToModels(modelArgs)
    addToModelsTest_FrequencyFilters(modelArgs)
    myAnn.updatePaths(outputPath = os.path.dirname(os.path.realpath(__file__)) + "/")

    testing = True
    if (testing):
        #use default data: input002, input142
        #data,labels,recordCount = 
        myData.readData()
        myData.expandDims()
        #data = data[:1000]
        print(myData.data.shape)
        #labels = labels[:1000]
        dataFiles = ",".join(inputData())
        cvFolds = 10
        valPerc = 0.10
        epochs = 10
        batchSize =int(((myData.record_count*(1-valPerc))/cvFolds)+1)
        with open("fileTrainTestParams.txt",'w') as params:
            params.write(f"dataFiles: {dataFiles}\ncvFolds: {cvFolds}\n")
            params.write(f"validation_split: {valPerc}\nepoch: {epochs}\n")
            params.write(f"batchSize: {batchSize}\n")
        #myAnn.buildModelStack(myData.data,myData.labels)
        #return
        myAnn.parameterSearch(modelArgs[:10],myData.data,myData.labels,valSplit=0.10)
    else:
        myData.readData(fnames=inputData())
        myData.expandDims()
        dataFiles = ",".join(inputData())
        cvFolds = 10
        valPerc = 0.10
        epochs = 10
        batchSize =int(((myData.record_count*(1-valPerc))/cvFolds)+1)
        with open("fileTrainTestParams.txt",'w') as params:
            params.write(f"dataFiles: {dataFiles}\ncvFolds: {cvFolds}\n")
            params.write(f"validation_split: {valPerc}\nepoch: {epochs}\n")
            params.write(f"batchSize: {batchSize}\n")
        myAnn.parameterSearch(modelArgs,myData.data,myData.labels,numSplits=cvFolds, valSplit=valPerc, epochs=epochs, batchSize=batchSize)

def inputData():
    #this is the entire list
    #return np.array("input001.csv,input002.csv,input011.csv,input012.csv,input031.csv,input032.csv,input041.csv,input042.csv,input081.csv,input082.csv,input091.csv,input101.csv,input112.csv,input142.csv,input151.csv,input152.csv,input161.csv,input162.csv,input171.csv,input172.csv".split-(","))
    #These choices were made by which ones had the most REM
    t = "input152.csv,input042.csv,input171.csv,input161.csv,input082.csv,input091.csv,input002.csv,input142.csv,input031.csv,input151.csv,input101.csv,input032.csv".split(",")
    return np.array(t[:max( min(numOfInputFiles,len(t)),2)]) 

def addToModelsTest_FrequencyFilters(modelArgs, addConvFilters=True):
    #low freq to high freq
    tempArgs = []
    convFilters = {
        66:[33],
        20:[33,13],
        10:[13,8],
        5 :[8,4],
        3 :[4]
    }
    kernalSizes = [3,5,10,20,66]
    numFilters = range(1,6)
    layerStartingSizeDividers = range(1,11)
    layerSizeDecreases = range(1,6)
    hiddenLayers = range(0,11)
    for kernalSize, numFilter, layerStartingSizeDivider, layerSizeDecrease, hiddenLayer in [(kernalSize, numFilter, layerStartingSizeDivider, layerSizeDecrease, hiddenLayer) for kernalSize in kernalSizes for numFilter in numFilters for layerStartingSizeDivider in layerStartingSizeDividers for layerSizeDecrease in layerSizeDecreases for hiddenLayer in hiddenLayers]:
        skip = layerSizeDecrease > max(2,hiddenLayer)
        skip = skip or layerStartingSizeDivider > max(2,numFilter)
        if (skip):
            continue
        tempArgs.append([])
        if (addConvFilters):
            for convFilter in convFilters[kernalSize]:
                tempArgs[-1].append({
                    'layer': 'conv1d',
                    'no_filters' : 1,
                    'kernal_size': convFilter,
                    'padding'    : 'same',
                    'activation' : 'relu'
                })
        tempArgs[-1].append({
            'layer': 'conv1d',
            'no_filters' : numFilter,
            'kernal_size': kernalSize,
            'padding'    : 'same',
            'activation' : 'relu'
        })
        
        layerSize = int((numFilter*3000)/layerStartingSizeDivider)
        for hid in range(hiddenLayer):
            
            tempArgs[-1].append({
                'layer': 'dense',
                'output': layerSize,
                'activation':'relu'
            })
            layerSize = int(layerSize/layerSizeDecrease)
            if (layerSize < 5):
                break
        tempArgs[-1].append({
            'layer': 'flatten'
        })
        tempArgs[-1].append({
            'layer': 'dense',
            'output': 2,
            'activation':'softmax'
        })
        tempArgs[-1].append({
            'layer': 'compile',
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics':['acc']
        })

    for i in np.random.randint(0,len(tempArgs),size=1000):
        modelArgs.append(tempArgs[i])
    

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

    if smallModelOnly:
        return
        
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

    #THIS ONE IS TO BIG TO TRAIN ON ORION00
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

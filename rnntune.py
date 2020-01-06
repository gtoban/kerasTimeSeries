import pandas as pd
import numpy as np
from kerasOOP import keras_ann, cnn_data
import os
import json


numOfInputFiles = 5
def main():
    print("Initializing")
    myAnn = keras_ann()
    myloc = os.path.expanduser('~') + "/kerasTimeSeries/"
    myData = cnn_data(dataPath= os.path.expanduser('~') + "/eegData/")

                   # 0        1              2              3
    useCandidate = ['', 'topTwo.csv', 'topTen.csv', 'candidate.csv'][0]
    testing = True
    optimizeOptimizer = False
    saveModel = False
    modelArgs = [] #getModels() small models only for now!
    #addToModels(modelArgs)
    print("Collecting Models")
    if (useCandidate == ''):
        addToModels(modelArgs)
    #else:
    #    getCandidates(modelArgs, fname=useCandidate, optimize = optimizeOptimizer) #
    print(f"Number of Models: {len(modelArgs)}")
    
    myAnn.updatePaths(outputPath = os.path.dirname(os.path.realpath(__file__)) + "/")
    
    
    
    if (testing):
        myData.readData()
    else:
        myData.readData(fnames=inputData())
    lowFreq=highFreq=None
    dataFiles = ",".join(inputData())
    cvFolds = 0 if saveModel else 10
    valPerc = 0.10
    epochs = 1 if saveModel else 100
    batchSize = 32 if saveModel else int(((myData.record_count*(1-valPerc))/cvFolds)+1)
    with open("fileTrainTestParams.txt",'w') as params:
        params.write(f"dataFiles: {dataFiles}\ncvFolds: {cvFolds}\n")
        params.write(f"validation_split: {valPerc}\nepoch: {epochs}\n")
        params.write(f"batchSize: {batchSize}\n")
        params.write(f"frequency: {lowFreq} - {highFreq}\n")
        params.write(f"normSTD  : {myData.normSTD}\n")
        params.write(f"normMean : {myData.normMean}")

    if (saveModel):
        myAnn.trainModel(modelArgs,myData.data,myData.labels, valSplit=valPerc, epochs=epochs, batchSize=batchSize, visualize=False, saveLoc=myloc)
        return
    if (testing):
        myAnn.parameterSearch(modelArgs[:1],myData.data,myData.labels,valSplit=0.10)
    else:
        myAnn.parameterSearch(modelArgs,myData.data,myData.labels,numSplits=cvFolds, valSplit=valPerc, epochs=epochs, batchSize=batchSize, saveModel=saveModel, visualize=False, saveLoc=myloc)


def addToModels(modelArgs):
    #low freq to high freq
    numOfModels = 100
    possibleRecurrentLayers = 5
    possibleUnits = [int(val) for val in np.linspace(5,600,num=25)]

    possibleDenseLayers = 5
    maxHiddenUnits = [10,30,50,70,100,120,150,170,200]
    hiddenUnitsDivisors = [1,2,3,4]
    denseActivations = ['sigmoid','relu','tanh']
    for i in range(numOfModels):
        numOfRecurrentLayers = int(np.random.randint(1,possibleRecurrentLayers))
        denseActivation = denseActivations[np.random.randint(len(denseActivations))]
        modelArgs.append([{
            'layer': 'input',
            'shape': (3,10)
        }])
        # randomly add a timeDistributed Dense layer here
        if int(np.random.randint(2)) == 1:
            modelArgs[-1].append({
                'layer': 'dense',
                'output': 10,
                'activation': denseActivation,
                'wrapper': 'timedistributed'
                })
        bidirectional = True if int(np.random.randint(2)) == 1 else False
        if numOfRecurrentLayers > 1:
            for dummy in range(numOfRecurrentLayers-1):
                units = possibleUnits[int(np.random.randint(len(possibleUnits)))]
                modelArgs[-1].append({
                    'layer': 'rnn',
                    'type' : 'gru',
                    'units': units,
                    'return_sequences': True
                    })
                bidirectional = True if int(np.random.randint(2)) == 1 else False
                if bidirectional:
                    modelArgs[-1][-1]['wrapper'] = 'bidirectional'
                # randomly add a timeDistributed Dense layer here
                if int(np.random.randint(2)) == 1:
                    modelArgs[-1].append({
                    'layer': 'dense',
                    'output': units*2 if bidirectional else units,
                    'activation': denseActivation,
                    'wrapper': 'timedistributed'
                    })
        units = possibleUnits[int(np.random.randint(len(possibleUnits)))]
        timeDist = True if int(np.random.randint(2)) == 1 else False
        bidirectional = True if int(np.random.randint(2)) == 1 else False
        modelArgs[-1].append({
            'layer': 'rnn',
            'type' : 'gru',
            'units': units,
            'return_sequences': timeDist
            })
        
        if bidirectional:
            modelArgs[-1][-1]['wrapper'] = 'bidirectional'
        if timeDist:
            modelArgs[-1].append({
                    'layer': 'dense',
                    'output': units*2 if bidirectional else units,
                    'activation': denseActivation,
                    'wrapper': 'timedistributed'
                    })
            modelArgs[-1].append({
                'layer': 'flatten'})
        divisor = hiddenUnitsDivisors[np.random.randint(len(hiddenUnitsDivisors))]
        output = maxHiddenUnits[np.random.randint(len(maxHiddenUnits))]
        
        for dummy in range(0,int(np.random.randint(possibleDenseLayers))):
            output = int(output//divisor) if dummy > 0 and output > 5 else output
            modelArgs[-1].append({
                'layer': 'dense',
                'output': output,
                'activation': denseActivation
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
def inputData():
    #this is the entire list
    #return np.array("input001.csv,input002.csv,input011.csv,input012.csv,input031.csv,input032.csv,input041.csv,input042.csv,input081.csv,input082.csv,input091.csv,input101.csv,input112.csv,input142.csv,input151.csv,input152.csv,input161.csv,input162.csv,input171.csv,input172.csv".split-(","))
    #These choices were made by which ones had the most REM
    t = "outinput152.csv,outinput042.csv,outinput171.csv,outinput161.csv,outinput082.csv,outinput091.csv,outinput002.csv,outinput142.csv,outinput031.csv,outinput151.csv,outinput101.csv,outinput032.csv".split(",")
    return np.array(t[:max( min(numOfInputFiles,len(t)),2)])

main()

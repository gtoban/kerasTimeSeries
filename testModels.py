import pandas as pd
import numpy as np
import json
import os
from os import listdir
from kerasOOP import keras_ann, ann_data
import sys
#NOTE: FLAG FOR TESTING SMALL MODELS ONLY!!
smallModelOnly = True
numOfInputFiles = 10
def main():
    if (len(sys.argv) < 2):
        freqBand = 'theta'
    else:
        freqBand = sys.argv[1]
    print("Initializing")
    myAnn = keras_ann()
    myloc = os.path.expanduser('~') + "/kerasTimeSeries/"
    weightPath=os.path.expanduser('~') + "/localstorage/kerasTimeSeries/myweights/" + freqBand + "/"
    myData = ann_data(dataPath= os.path.expanduser('~') + "/eegData/")
    
    modelArgs = [] 
    print("Collecting Models")
    getCandidates(modelArgs, fname=weightPath+"topTwo.csv", optimize = False)
    weights = []
    getWeights(weights,weightPath)
    [normSTD, normMean] = getNorm(weightPath)
    [lowFreq, highFreq, _] = ann_data.getFreqBand(freqBand)
    #print(modelArgs)
    #print(weights)
    #return
    
    testing = False
    if (testing):
        myData.readData()
    else:
        myData.readData(fnames=inputData())
    myData.filterFrequencyRange(low=lowFreq, high=highFreq)
    myData.expandDims()
    myData.normalize(normSTD=normSTD, normMean=normMean)
    
    myAnn.testModel(modelArgs,myData.data,myData.labels,weights=weights,loadLoc=weightPath)


def getNorm(myDir):
    normSTD = normMean = None
    for filename in listdir(myDir):
        if 'fileTrainTestParams.txt' in filename:
            with open(myDir + filename) as pfile:
                for line in pfile:
                    if ('normSTD' in  line):
                        normSTD = float(line.split(':')[1])
                    elif ('normMean' in line):
                        normMean = float(line.split(':')[1])
                        
            return [normSTD, normMean]
    
def getWeights(weights,myDir):

    currentID=''
    ids = []
    cidList = {}
    for filename in listdir(myDir):        
        if ('.h5' in filename):
            tid = int(filename.split('.')[0])
            if (tid not in ids):
                ids.append(tid)
                cidList[tid] = []
            cidList[tid].append(filename)
    for myid in sorted(ids):
        weights.append(cidList[myid])
        
        
def inputData():
    #this is the entire list
    #return np.array("input001.csv,input002.csv,input011.csv,input012.csv,input031.csv,input032.csv,input041.csv,input042.csv,input081.csv,input082.csv,input091.csv,input101.csv,input112.csv,input142.csv,input151.csv,input152.csv,input161.csv,input162.csv,input171.csv,input172.csv".split-(","))
    #These choices were made by which ones had the most REM
    t = "input152.csv,input042.csv,input171.csv,input161.csv,input082.csv,input091.csv,input002.csv,input142.csv,input031.csv,input151.csv,input101.csv,input032.csv".split(",")
    return np.array(t[:max( min(numOfInputFiles,len(t)),2)]) 

def getCandidates(modelArgs, fname="candidate.csv", optimize = False):
    
    for index, candidate in pd.read_csv(fname, sep='|').iterrows():
        modelArgs.append(json.loads(candidate["model"]))

    if (optimize):
        numOfCandidates = len(modelArgs)
        for cid in range(numOfCandidates):
            c = modelArgs[cid]
            for layer in c:
                if (layer["layer"] == 'compile'):
                    #['adam','sgd','rmsprop','nadam']
                    if (layer['optimizer'] == 'sgd'):
                        addSGD(modelArgs, c, int(100/numOfCandidates))
                    
                    if (layer['optimizer'] == 'adam'):
                        addAdam(modelArgs, c, int(100/numOfCandidates))

                    if (layer['optimizer'] == 'nadam'):
                        addNAdam(modelArgs, c, int(100/numOfCandidates))

                    if (layer['optimizer'] == 'rmsprop'):
                        addRMSprop(modelArgs, c, int(100/numOfCandidates))
def addAdam(modelArgs, tmodel, numKeepIndexes):
    target = tmodel
    learningRates = [0.1,0.01,0.001,0.0001]
    beta1s = [0.98,0.99,0.999,0.9999]
    beta2s = [0.98,0.99,0.999,0.9999]
    amsgrads = [True,False]
    total = len(learningRates) * len(beta1s) * len(beta2s) * len(amsgrads)
    ci = 0
    for i in range(len(target)):
        if (target[i]['layer'] == 'compile'):
            ci = i
    first = True
    keepIndexes = np.concatenate((np.array([0]),np.random.randint(1,total,size=numKeepIndexes-1)))
    index = 0
    for lr, beta1, beta2, amsgrad in [(lr, beta1, beta2, amsgrad) for lr in learningRates for beta1 in beta1s for beta2 in beta2s for amsgrad in amsgrads]:
        if (index not in keepIndexes):
            index += 1
            continue
        target[ci]['optimizerOptions'] = [lr, beta1, beta2, amsgrad]
        if (not first):
            modelArgs.append(target)
        first = False
        target = tmodel.copy()
        index += 1

def addNAdam(modelArgs, tmodel, numKeepIndexes):
    target = tmodel
    learningRates = [0.1,0.01,0.001,0.0001]
    beta1s = [0.98,0.99,0.999,0.9999]
    beta2s = [0.98,0.99,0.999,0.9999]
    
    total = len(learningRates) * len(beta1s) * len(beta2s) * len(amsgrads)
    ci = 0
    for i in range(len(target)):
        if (target[i]['layer'] == 'compile'):
            ci = i
    first = True
    keepIndexes = np.concatenate((np.array([0]),np.random.randint(1,total,size=numKeepIndexes-1)))
    index = 0
    for lr, beta1, beta2  in [(lr, beta1, beta2) for lr in learningRates for beta1 in beta1s for beta2 in beta2s]:
        if (index not in keepIndexes):
            index += 1
            continue
        target[ci]['optimizerOptions'] = [lr, beta1, beta2]
        if (not first):
            modelArgs.append(target)
        first = False
        target = tmodel.copy()
        index += 1

def addSGD(modelArgs,tmodel, numKeepIndexes):
    target = tmodel
    learningRates = [0.1,0.01,0.001,0.0001]
    momentums = [0.0,0.1,0.3,0.5,0.7,0.9,0.99]
    nesterovs = [True,False]
    total = len(learningRates) * len(momentums) * len(nesterovs)
    ci = 0
    for i in range(len(target)):
        if (target[i]['layer'] == 'compile'):
            ci = i
    first = True
    keepIndexes = np.concatenate((np.array([0]),np.random.randint(1,total,size=numKeepIndexes-1)))
    index=0
    for lr, mo, nesterov in [(lr, mo, nesterov) for lr in learningRates for mo in momentums for nesterov in nesterovs]:
        if (index not in keepIndexes):
            index += 1
            continue
        target[ci]['optimizerOptions'] = [lr, mo, nesterov]
        if (not first):
            modelArgs.append(target)
        first = False
        target = tmodel.copy()
        index += 1

def addRMSprop(modelArgs,tmodel, numKeepIndexes):
    target = tmodel
    learningRates = [0.1,0.01,0.001,0.0001]
    rhos = [0.0,0.1,0.3,0.5,0.7,0.9,0.99]
    
    total = len(learningRates) * len(rhos)
    ci = 0
    for i in range(len(target)):
        if (target[i]['layer'] == 'compile'):
            ci = i
    first = True
    keepIndexes = np.concatenate((np.array([0]),np.random.randint(1,total,size=numKeepIndexes-1)))
    index=0
    for lr, rho in [(lr, rho) for lr in learningRates for rho in rhos]:
        if (index not in keepIndexes):
            index += 1
            continue
        target[ci]['optimizerOptions'] = [lr, rho]
        if (not first):
            modelArgs.append(target)
        first = False
        target = tmodel.copy()
        index += 1
        
def addToModelsTest_FrequencyFilters(modelArgs, addConvFilters=True, manyFilters = False , numKeepIndexes = 1000, kernalPreset=-1):
    useStartingDividers = False
    #low freq to high freq
    convFilters = {
        66:[33],
        20:[33,13],
        10:[13,8],
        5 :[8,4],
        3 :[4]
    }
    if kernalPreset < 1:
        kernalSizes = [3,5,10,20,66]
    else:
        kernalSizes = [kernalPreset]
    if (manyFilters):
        numFilters = [20,50,100,200,500]
    else:    
        numFilters = range(5,11,5) #use 5 and 10 only
    if (useStartingDividers):
        layerSizeStarters = range(1,11)
    else:   
        layerSizeStarters = [800] #[10,50,100,200,400,800]
    layerSizeDecreases = [1]#range(1,6)
    hiddenLayers = [10]#range(1,11)
    poolTypes = ['maxpool1d','avgpool1d',None]
    poolSizes = [2,4,8,12,24]    
    strideDownscaleFactors = [1,2,3,4,None]
    activationFunctions = ['relu','tanh','sigmoid']
    kernelInitializers = ['zeros','ones','random_normal','random_uniform', 'glorot_normal','glorot_uniform','he_normal','he_uniform']
    biasInitializers = ['zeros','ones','random_normal','random_uniform', 'glorot_normal','glorot_uniform','he_normal','he_uniform']
    optimizers = ['adam','sgd','rmsprop','nadam']

    totalSize = len(kernalSizes) * len(numFilters) * len(layerSizeStarters)
    totalSize *= len(layerSizeDecreases) * len(hiddenLayers)
    totalSize *= len(poolTypes) * len(poolSizes) * len(strideDownscaleFactors)
    print(f"totalSize: {totalSize}")
    keepIndexes = np.random.randint(0,totalSize,size=numKeepIndexes)
    index = 0
    for kernalSize, numFilter, layerSizeStarter, layerSizeDecrease, hiddenLayer, poolType, poolSize, strideDownscaleFactor in [(kernalSize, numFilter, layerSizeStarter, layerSizeDecrease, hiddenLayer, poolType, poolSize, strideDownscaleFactor) for kernalSize in kernalSizes for numFilter in numFilters for layerSizeStarter in layerSizeStarters for layerSizeDecrease in layerSizeDecreases for hiddenLayer in hiddenLayers for poolType in poolTypes for poolSize in poolSizes for strideDownscaleFactor in strideDownscaleFactors]:
        #skip = layerSizeDecrease > max(2,hiddenLayer)
        skip = poolSize > kernalSize
        skip = skip and (numFilter > 20 and poolType is None)
        if (skip):
            continue
        if (index not in keepIndexes):
            index += 1
            continue
        activation = activationFunctions[int(np.random.randint(0,len(activationFunctions)))]
        kernelInitializer = kernelInitializers[int(np.random.randint(0,len(kernelInitializers)))]
        biasInitializer = biasInitializers[int(np.random.randint(0,len(biasInitializers)))]
        optimizer = optimizers[int(np.random.randint(0,len(optimizers)))]
        modelArgs.append([])
        if (addConvFilters):
            for convFilter in convFilters[kernalSize]:
                modelArgs[-1].append({
                    'layer': 'conv1d',
                    'no_filters' : 1,
                    'kernal_size': convFilter,
                    'padding'    : 'same',
                    'activation' : activation,
                    'kernel_initializer': kernelInitializer,
                    'bias_initializer': biasInitializer
                })
        modelArgs[-1].append({
            'layer': 'conv1d',
            'no_filters' : numFilter,
            'kernal_size': kernalSize,
            'padding'    : 'same',
            'activation' : activation,
            'kernel_initializer': kernelInitializer,
            'bias_initializer': biasInitializer
        })

        if (poolType is not None):
            modelArgs[-1].append({
                'layer': poolType,
                'pool_size': poolSize,
                'strides':strideDownscaleFactor,
                'padding':'valid'
            })
        modelArgs[-1].append({
            'layer': 'flatten'
        })
        if (useStartingDividers):
            layerSize = int((numFilter*3000)/layerSizeStarter)
        else:
            layerSize = layerSizeStarter
        for hid in range(hiddenLayer):
            
            modelArgs[-1].append({
                'layer': 'dense',
                'output': layerSize,
                'activation':activation,
                'kernel_initializer': kernelInitializer,
                'bias_initializer': biasInitializer
            })
            layerSize = int(layerSize/layerSizeDecrease)
            if (layerSize < 5):
                break
        
        modelArgs[-1].append({
            'layer': 'dense',
            'output': 2,
            'activation':'softmax',
            'kernel_initializer': kernelInitializer,
            'bias_initializer': biasInitializer
        })
        
        modelArgs[-1].append({
            'layer': 'compile',
            'optimizer': optimizer,
            'loss': 'categorical_crossentropy',
            'metrics':['acc']
        })
        index += 1
        

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

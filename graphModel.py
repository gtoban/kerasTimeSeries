import pandas as pd
import numpy as np
import json
import os
from os import listdir
from kerasOOP import keras_ann, ann_data
from modelBuilder import ModelBuilder
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
    mybuild = ModelBuilder()
    myloc = os.path.expanduser('~') + "/localstorage/kerasTimeSeries/"
    weightPath=os.path.expanduser('~') + "/localstorage/kerasTimeSeries/myweights/" + freqBand + "/"
    print("Collecting Models")
    
    weights = []
    modelArgs = []
    print("GET PARAMS")
    mybuild.getCandidates(modelArgs, fname=weightPath+"topTwo.csv", optimize = False)
    myAnn.getWeights(weights,weightPath)

    #=============
    # for collecting data
    #=============
    print("GET PARAMS")
    myData = ann_data(dataPath= os.path.expanduser('~') + "/eegData/")
    [normSTD, normMean] = myAnn.getNorm(weightPath)
    [lowFreq, highFreq, _] = ann_data.getFreqBand(freqBand)
    testing = True
    if (testing):
        myData.readData()
    else:
        pass#myData.readData(fnames=inputData())
    #myData.filterFrequencyRange(low=lowFreq, high=highFreq)
    myData.expandDims()
    myData.normalize(normSTD=normSTD, normMean=normMean)
    
    #print(weights)
    print("PRINT MODEL")
    myAnn.printModel(modelArgs, weights=weights,printLoc=myloc, loadLoc=weightPath,X=myData.data,Y=myData.labels)
    print("DONE")

main()

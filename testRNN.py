import pandas as pd
import numpy as np
from kerasOOP import keras_ann, cnn_data
import os
import json
from os import listdir


numOfInputFiles = 5
def main():
    print("Initializing")
    myAnn = keras_ann()
    myloc = os.path.expanduser('~') + "/kerasTimeSeries/myweights/rnn"
    myloc = '/nfshome/gst2d/localstorage/kerasTimeSeries/myweights/rnn'
    myData = cnn_data(dataPath= os.path.expanduser('~') + "/eegData/")
    tloc = myloc
    myloc += '/'
    for fname in listdir(myloc):
        if ('fileModel' in fname and '~' not in fname):
            useCandidate = myloc+str(fname)
    testing = True
    modelArgs = [] #getModels() small models only for now!
    #addToModels(modelArgs)
    print(useCandidate)
    mod = pd.read_csv(useCandidate, sep='|', header=0)
    print(mod.columns)
    for index, candidate in pd.read_csv(useCandidate, sep='|', header=0).iterrows():
        
        modelArgs.append(json.loads(candidate["model"]))
    
    print(f"Number of Models: {len(modelArgs)}")

    weights = []
    myAnn.getWeights(weights,myloc)
    print(weights)
    myAnn.updatePaths(outputPath = os.path.dirname(os.path.realpath(__file__)) + "/")
    
    
    
    myData.readData(fnames=inputData())
    myAnn.testModel(modelArgs,myData.data,myData.labels,weights=weights,loadLoc=myloc)

def inputData():
    #this is the entire list
    #return np.array("input001.csv,input002.csv,input011.csv,input012.csv,input031.csv,input032.csv,input041.csv,input042.csv,input081.csv,input082.csv,input091.csv,input101.csv,input112.csv,input142.csv,input151.csv,input152.csv,input161.csv,input162.csv,input171.csv,input172.csv".split-(","))
    #These choices were made by which ones had the most REM
    t = "outinput152.csv,outinput042.csv,outinput171.csv,outinput161.csv,outinput082.csv,outinput091.csv,outinput002.csv,outinput142.csv,outinput031.csv,outinput151.csv,outinput101.csv,outinput032.csv".split(",")
    return np.array(t[:max( min(numOfInputFiles,len(t)),2)])

main()

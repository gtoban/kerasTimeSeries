import numpy as np
import json
from os import listdir, path
from kerasOOP import ann_data, keras_ann
def main():
    # load models and weights
    print("Initializing")
    myAnn = keras_ann()
    mySaveLoc = path.expanduser('~') + "/eegData/"
    myLoadLoc = path.expanduser('~') + "/localstorage/kerasTimeSeries/myweights/"
    myData = ann_data(dataPath= path.expanduser('~') + "/eegData/")
    allWeights = []
    models = []
    freqs = ['delta','theta','alpha','beta1','beta2']

    # there are several models to choose from,
    # manually choose the index of the model 
    modelChoice = {}
    for freq in freqs:
        modelChoice[freq] = 0
    modelChoice['delta'] = 1
    
    for freq in freqs:
        weights = []
        myAnn.getWeights(weights,myLoadLoc + freq)
        allWeights.append(myLoadLoc + freq + '/' + weights[modelChoice[freq]][0])
        #print(weights)
        #print(weights[modelChoice[freq]][0])
        for filename in listdir(myLoadLoc + freq):
            if 'Model' in filename:
                with open(myLoadLoc + freq + '/' + filename) as tfile:
                    modelIdtemp = 0
                    while modelIdtemp < modelChoice[freq]:
                        tfile.readline()
                        modelIdtemp += 1
                    #print(f"model: {modelIdtemp}")
                    models.append(json.loads(tfile.readline().split('|')[-1].strip()))
    
                    
                    
            
        #print(allWeights[-1], models[-1], '\n')
                    
                    
    t = "input081.csv,input091.csv,input011.csv,input162.csv,input002.csv,input171.csv,input151.csv,input031.csv,input041.csv,input152.csv,input142.csv,input101.csv,input042.csv,input012.csv,input032.csv,input112.csv,input161.csv,input001.csv,input082.csv,input172.csv".split(",")
    i = 0
    for inputfile in t:
        print()
        print('='*16)
        print(inputfile)
        myData.readData(fnames=[inputfile])
        [normSTD, normMean] = myAnn.getNorm(myLoadLoc + freq + '/')
        myData.expandDims()
        myData.normalize(normSTD=normSTD, normMean=normMean)
        myAnn.saveModelOutput(models,myData.data,myData.labels, weights=allWeights, saveLoc=mySaveLoc,saveName='out' + inputfile, loadLoc='')
        i += 1
    # for every sourceFile
    # read file
    # convert to convData


main()

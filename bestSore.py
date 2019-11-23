import pandas as pd
import numpy as np
import sys
import json
from os import walk, path, listdir

def main():
    if (len(sys.argv) < 3):
        print("test ID range [min max) required")
        exit(0)
    try:
        amin = int(sys.argv[1])
        amax = int(sys.argv[2])
        if (amin >= amax):
            t = amin
            amin = amax
            amax = amin
    except Exception as e:
        print("Did not enter integer ranges like this: min max")
        exit(0)
    
    myDir = path.expanduser('~') + "/localstorage/kerasTimeSeries/modelResults"
    if (path.isdir(myDir)):
        myDir += "/"
    else:
        myDir = path.expanduser('~')
    dataColumns = ['modelNum', 'True REM', 'False REM', 'False NonREM', 'True NonREM','Acc', 'Sens', 'Spec', 'Recall', 'Precision', 'f1score', 'finalLoss','f1Avg', 'zeroSens', 'zeroSpec', 'model', 'testID']
    candidates = pd.DataFrame(columns=dataColumns)
    #["testID","modelNum",'f1Avg',"model"]
    for id in range(amin,amax):
        idNum = f"{int(id):05d}"
        for filename in listdir(myDir):
            if (idNum in filename):
                if ("Result" in filename):
                    #False NonREM|True NonREM|Acc|Sens|Spec
                    results = pd.read_csv(myDir + filename, sep='|')#, usecols=['modelNum','True REM','False REM'])
                #print(results['True REM'] > 0
                    results['f1Avg'] = results[['modelNum','f1score']].groupby('modelNum').transform(np.average)
                    results['zeroSens']= results[['modelNum','Sens']].groupby('modelNum').transform(np.prod)
                    results['zeroSpec']= results[['modelNum','Spec']].groupby('modelNum').transform(np.prod)
                    results = results[results['modelNum'].isin(results[(results['zeroSens']>0) & (results['zeroSpec'] >0)]['modelNum'])]
                    #results = results[(results["False REM"] > 0) | (results["True REM"] > 0)]
                if ("Model" in filename):
                    models = pd.read_csv(myDir + filename, sep="|", header=None,names=['modelNum','model'])
                    
        data = pd.merge(results,models,on='modelNum', how='inner')
        data = data.assign(testID=idNum)
        #print(data.columns)
        
        candidates = candidates.append(data, ignore_index=True)

    candidates['f1Avg'] = candidates[['model','f1score']].groupby('model').transform(np.average)
    candidates['f1Med'] = candidates[['model','f1score']].groupby('model').transform(np.median)
    candidates['specAvg'] = candidates[['model','Spec']].groupby('model').transform(np.average)
    candidates['sensAvg'] = candidates[['model','Sens']].groupby('model').transform(np.average)
    candidates.drop_duplicates(subset=["model"],inplace=True)
    candidates=candidates.sort_values(by=['f1Avg'], ascending=False)
    print(candidates[["testID","modelNum","f1Avg","model"]])
    print(candidates.iloc[0]["model"])
    
    print(f"testID|modelNum|f1Avg|f1Med|Spec NR|Sens R|kernelInit|biasInit|optimizer|optoptions|kernelSize|numKernels|activation|pool|numDense|sizes")
    print(f"testID|modelNum|f1Avg|f1Med|Spec NR|Sens R|kernelInit|biasInit|optimizer|optoptions|lstms|units|bidirectional|numDense|sizes")
    for index, row in candidates.iterrows():
        modelInfo = json.loads(row['model'])
        modelNum = row['modelNum']
        testID = row['testID']
        spec = float(row['specAvg'])
        sens = float(row['sensAvg'])
        f1Avg = row['f1Avg']
        f1Med = row['f1Med']
        kernelSize = 0
        numKernels = 0
        activation = ""
        pool = 'No'
        numDense = 0
        denseNodes = []
        kernelInit = 'glorot_uniform'
        biasInit = 'zeros'
        optimizer = 'adam'
        optoptions = 'none'
        units = 0
        lstms = 0
        bidirectional = 'false'
        for layer in modelInfo:
            if (layer['layer'] == 'conv1d'):
                numKernels = int(layer['no_filters'])
                kernelSize = int(layer['kernal_size'])
                activation = layer['activation']
                try:
                    kernelInit = layer['kernel_initializer']
                    biasInit = layer['bias_initializer']
                except:
                    pass
            elif (layer['layer'] == 'dense'):
                numDense += 1
                denseNodes.append(layer['output'])
            elif ('pool' in layer['layer']):
                pool = layer['layer']
            elif('compile' in layer['layer']):
                try:
                    optimizer = layer['optimizer']
                except:
                    pass
                try:
                    optoptions = ','.join([ str(opt) for opt in layer['optimizerOptions']])
                except:
                    optoptions = 'none'
            elif ('lstm' in layer['layer']):
                lstms += 1
                units = layer['units']
                try:
                    if (layer['wrapper'] == 'bidirectional'):
                        bidirectional = 'true'
                except:
                    pass
                        
                    
        #print(f"{testID}|{modelNum:3d}|{f1Avg:.3f}|{f1Med:.3f}|{spec:.3f}|{sens:.3f}|{kernelInit}|{biasInit}|{optimizer}|{optoptions}|{kernelSize:3d}|{numKernels:4d}|{activation:5s}|{pool:3s}|{numDense:3d}|" + ",".join([f"{size}" for size in denseNodes]))
        print(f"{testID}|{modelNum:3d}|{f1Avg:.3f}|{f1Med:.3f}|{spec:.3f}|{sens:.3f}|{kernelInit}|{biasInit}|{optimizer}|{optoptions}|{lstms}|{units}|{bidirectional}|{numDense:3d}|" + ",".join([f"{size}" for size in denseNodes]))
    topten = candidates.iloc[:4]
    topten[["testID","modelNum","model"]].to_csv("topTwo.csv",sep="|",index=False,quoting=3) #csv.QUOTE_NONE
main()

import pandas as pd
import numpy as np
import sys
import json
from os import walk, path, listdir

def main():
    if (len(sys.argv) < 2):
        print("id number required")
        exit(0)
    idNum = f"{int(sys.argv[1]):05d}"
    myDir = path.expanduser('~') + "/localstorage/kerasTimeSeries/modelResults"
    if (path.isdir(myDir)):
        myDir += "/"
    else:
        myDir = path.expanduser('~')

    
    for filename in listdir(myDir):
        if (idNum in filename):
            if ("Result" in filename):
                results = pd.read_csv(myDir + filename, sep='|', usecols=['modelNum','True REM','False REM'])
                #print(results['True REM'] > 0
                results = results[(results['True REM']>0) | (results['False REM'] >0)]
            if ("Model" in filename):
                models = pd.read_csv(myDir + filename, sep="|", header=None,names=['modelNum','model'])
            #print(filename)
            #return
    #models = models[models['modelNum'] in results['modelNum']]
    data = pd.merge(results,models,on='modelNum', how='inner')
    data[['tp_sum','tn_sum']] = data[['modelNum','True REM','False REM']].groupby('modelNum').transform(np.sum)
    data['t_ratio'] = np.divide(data['tp_sum'],data['tn_sum']+np.finfo(float).eps)
    data=data.sort_values(by=['t_ratio','modelNum'], ascending=False)
    
    print(f"modelNum|tp|fp |tratio|kernelInit|biasInit|optimizer|kernelSize|numKernels|activation|pool|numDense|sizes")
    for index, row in data.iterrows():
        modelInfo = json.loads(row['model'])
        modelNum = row['modelNum']
        tp = int(row['True REM'])
        fp = int(row['False REM'])
        tratio = row['t_ratio']
        kernelSize = 0
        numKernels = 0
        activation = ""
        pool = 'No'
        numDense = 0
        denseNodes = []
        kernelInit = 'glorot_uniform'
        biasInit = 'zeros'
        optimizer = 'adam'
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
        print(f"{modelNum:3d}|{tp:3d}|{fp:3d}|{tratio:.3f}|{kernelInit}|{biasInit}|{optimizer}|{kernelSize:3d}|{numKernels:4d}|{activation:5s}|{pool:3s}|{numDense:3d}|" + ",".join([f"{size}" for size in denseNodes]))
        


main()

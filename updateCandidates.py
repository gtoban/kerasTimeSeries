import pandas as pd
import numpy as np
import sys
import json
from os import walk, path, listdir

def main():
    if (len(sys.argv) < 2):
        print("test ID required")
        exit(0)
    elif (len(sys.argv) < 3):

        amin = amax = -1
        idNum = f"{int(sys.argv[1]):05d}"
    else:
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

    candidates = pd.read_csv("candidate.csv", sep="|")
    print(candidates)
    if (amin == -1):
        candidates = candidates[candidates['model'].isin(getPassingCandidates(myDir, idNum))]
        
    else:
        for tid in range(amin,amax):
            idNum = f"{int(sys.argv[1]):05d}"
            candidates = candidates[candidates['model'].isin(getPassingCandidates(myDir, idNum))]
    print(candidates)
    #candidates.to_csv("candidate.csv",sep="|",index=False,quoting=3)
        
                                
        
        

def getPassingCandidates(myDir, idNum):
    for filename in listdir(myDir):
        if (idNum in filename):
            #False NonREM|True NonREM|Acc|Sens|Spec
            if ("Result" in filename):
                    
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
    data.drop_duplicates(subset=["modelNum"],inplace=True)
    return data['model']
        
main()

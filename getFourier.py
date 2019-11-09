import pandas as pd
import numpy as np
import os
import json
from kerasOOP import keras_ann, ann_data
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
#import progressbar

numOfInputFiles = 5
def main():
    myloc = os.path.expanduser('~') + "/kerasTimeSeries/"
    myData = ann_data(dataPath= os.path.expanduser('~') + "/eegData/")

    testing = True
    if (testing):
        myData.readData()
    else:
        myData.readData(fnames=inputData())
    #myData.filterFrequencyRange(low=2, high=45, order=8)
    fourier = np.array(myData.fourierAll())
    npnts = fourier.shape[1]
    #fourier = np.abs(fourier/npnts)**2
    #fourier = np.abs(fourier)**2
    topHowMany = fourier.shape[1]
    n_clusters = 5
    getTopFreqs(myData, fourier, topHowMany = topHowMany, label = None, freqLims = [2,35], n_clusters = n_clusters)
    for label in myData.indToLab:
        getTopFreqs(myData, fourier, topHowMany = topHowMany, label = label, freqLims = [2,35], n_clusters = n_clusters)
    
    
    #print(fourier.shape)
    #print(fourier[0])
    #mylegend = []
    ##print(myData.labels[0])
    ##print(myData.data[0][:30])
    ##return
    #
    #for i in range(5):
    #    plt.figure()
    #    plt.plot(hz,np.abs(fourier[100]/npnts)**2)
    #    #plt.figure()
    ##plt.figure()
    ##for i in range(5):
    ##    plt.plot(myData.data[i])
    ##    #mylegend.append('R' if myData.labels[i][0] == 1 else 'NR')
    ##plt.legend(mylegend)
    #plt.show()

def inputData():
    #this is the entire list
    #return np.array("input001.csv,input002.csv,input011.csv,input012.csv,input031.csv,input032.csv,input041.csv,input042.csv,input081.csv,input082.csv,input091.csv,input101.csv,input112.csv,input142.csv,input151.csv,input152.csv,input161.csv,input162.csv,input171.csv,input172.csv".split-(","))
    #These choices were made by which ones had the most REM
    t = "input152.csv,input042.csv,input171.csv,input161.csv,input082.csv,input091.csv,input002.csv,input142.csv,input031.csv,input151.csv,input101.csv,input032.csv".split(",")
    return np.array(t[:max( min(numOfInputFiles,len(t)),2)]) 

def getTopFreqs(data, tfourier, topHowMany = 5, label = None, freqLims = [None,None], n_clusters = 5):
    
    if label is not None:
        indices = np.where(data.obsLabels == data.labToInd[label])
        fourier = np.copy(tfourier[indices])
        #hz = hz[indices]
        print(f"for label: {label}")
    else:
        fourier = np.copy(tfourier)
        print(f"for all labels")

    npnts = tfourier.shape[1]
    srate = 100
    hz = np.linspace(0,srate//2,npnts)
    topFreqs = np.zeros(topHowMany*fourier.shape[0])
    low = 0
    high = 0
    print("Combining Frequencies")
    #bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    count = 0
    for f in fourier:
        if freqLims[0] is None and freqLims[1] is None:
            a = f
            thz = hz
        elif freqLims[1] is None:
            indices = np.where(hz > freqLims[0])
            a = f[indices]
            thz = hz[indices]
        elif freqLims[0] is None:
            indices = np.where(hz < freqLims[1])
            a = f[indices]
            thz = hz[indices]
        else:
            indices = np.where(hz > freqLims[0]) 
            a = f[indices]
            thz = hz[indices]
            indices = np.where(thz < freqLims[1])
            a = a[indices]
            thz = thz[indices]
        topmost = min(topHowMany,a.shape[0])
        high += topmost
        if topmost > 0:
            if topmost == thz.shape[0]:
                topFreqs[low:high] = thz
            else:
                topFreqs[low:high] = thz[np.argsort(a)[-topmost:]]
            #topFreqs.append(thz[np.argsort(a)[-topmost:]])
        low += topmost
        count += 1
        #bar.update(count)

    print("clustering")
    kmeans = KMeans(n_clusters=n_clusters).fit(np.array(topFreqs).reshape((-1,1)))
    print(kmeans.cluster_centers_)
    print(stats.describe(topFreqs))
    
    


    
main()

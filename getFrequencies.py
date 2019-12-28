import numpy as np

def main():
    print("Initializing")
    myloc = os.path.expanduser('~') + "/kerasTimeSeries/"
    myData = ann_data(dataPath= os.path.expanduser('~') + "/eegData/")

    myAnn.updatePaths(outputPath = os.path.dirname(os.path.realpath(__file__)) + "/")
    
    
    testing = True
    if (testing):
        myData.readData()
    else:
        myData.readData(fnames=inputData())
    fourier = np.ndarray(myData.fourierAll())
    print (fourier.shape)
    print(fourier[0])

main()

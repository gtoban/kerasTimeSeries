import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, AveragePooling1D, Input, Concatenate, Embedding,LSTM, GRU, TimeDistributed, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import backend as K, metrics, optimizers
from tensorflow.keras.utils import  plot_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix
from scipy import signal as scisig
from scipy import fftpack
import signal
import sys
from os import listdir

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True
    
class cnn_data(object):
    def __init__(self,dataPath="",outputPath=""):
        self.dataPath = dataPath
        self.outputPath = outputPath
        self.normSTD = 1
        self.normMean = 1
        
    def readData(self, fnames=["outinput002.csv","outinput142.csv"]):
        
        print("Starting Read")
        self.record_count = 0
        for fname in fnames:
            with open(self.dataPath + fname) as f:
                for line in f: 
                    if (line.strip()):
                        self.record_count += 1
        self.record_count -= 2
        sample = 0
        # records use
        # 3 eeg epochs in time each with
        # 5 algorithms producing
        # 2 predictions for REM vs NonREM
        self.data = np.zeros((self.record_count,3,5*2))
        self.labels = np.zeros((self.record_count,2))
        self.obsLabels = np.zeros((self.record_count))
        obsLabelsCount = {}
        for fname in fnames:
            

            RVNR = [0,0] #RVNR REM Vs NonREM
            f = open(self.dataPath + fname)
            lines = []
            lines.append(f.readline().strip())
            lines.append(f.readline().strip())
            lines.append(f.readline().strip())
            while(lines[2]):
                timeCount = 0
                for line in lines:
                    arow = line.split("|")
                    measure_count = 0
                    for ame in arow[1:]:
                        self.data[sample][timeCount][measure_count] = float(ame)
                        measure_count += 1
                    timeCount += 1
                    # use middle time for label
                    if timeCount == 1:
                        self.obsLabels[sample] = arow[0]
                        if arow[0] in obsLabelsCount.keys():
                            obsLabelsCount[arow[0]] += 1
                        else:
                            obsLabelsCount[arow[0]] = 1
                        self.labels[sample][int(arow[0])] = 1
                        RVNR[int(arow[0])] += 1
                
    
                sample += 1
                lines[0] = lines[1]
                lines[1] = lines[2]
                lines[2] = f.readline().strip()
            f.close()
            print(f"{fname} -> REM: {RVNR[0]}; NonREM: {RVNR[1]}")
            for l in obsLabelsCount:
                print(f"{l} : {obsLabelsCount[l]}")
               
        #return self.data, self.labels, self.record_count
    
class ann_data(object):
    def __init__(self,dataPath="",outputPath=""):
        self.dataPath = dataPath
        self.outputPath = outputPath
        self.normSTD = 1
        self.normMean = 1
        self.labToInd = {'W':0,'1':1, '2':2, '3': 3, '4': 4, 'R':5}
        self.indToLab = ['W','1', '2', '3', '4', 'R']
        
    def readData(self, fnames=["input002.csv","input142.csv"]):
        
        print("Starting Read")
        self.record_count = 0
        for fname in fnames:
            with open(self.dataPath + fname) as f:
                for line in f: 
                    if (line.strip()):
                        self.record_count += 1
        sample = 0
        self.data = np.zeros((self.record_count,3000))
        self.labels = np.zeros((self.record_count,2))
        self.obsLabels = np.zeros((self.record_count))
        obsLabelsCount = {}
        for fname in fnames:
            

            RVNR = [0,0] #RVNR REM Vs NonREM
            f = open(self.dataPath + fname)
            line = f.readline().strip()
            while(line):
                
                
                arow = line.split(",")
                self.obsLabels[sample] = self.labToInd[arow[0]]
                if arow[0] in obsLabelsCount.keys():
                    obsLabelsCount[arow[0]] += 1
                else:
                    obsLabelsCount[arow[0]] = 1
                self.labels[sample][0 if arow[0] == 'R' else 1] = 1
                RVNR[0 if arow[0] == 'R' else 1] += 1
                measure_count = 0
                for ame in arow[1:]:
                    self.data[sample][measure_count] = float(ame)
                    measure_count += 1
    
                sample += 1
                line = f.readline().strip()
            f.close()
            print(f"{fname} -> REM: {RVNR[0]}; NonREM: {RVNR[1]}")
            for l in obsLabelsCount:
                print(f"{l} : {obsLabelsCount[l]}")
               
        #return self.data, self.labels, self.record_count

    def expandDims(self):
        print("Expand Dims")
        self.data = np.expand_dims(self.data,axis=2)
        
        print("shape:", self.data.shape)

    def normalize(self, normSTD = None, normMean = None):
        print("Normalize")
        if (normSTD is None):
            self.normSTD = np.std(self.data)
            self.normMean = np.mean(self.data)
        else:
            self.normSTD = normSTD
            self.normMean = normMean
        self.data = np.divide(np.subtract(self.data,self.normMean), self.normSTD)
    def getFreqBand(freqBand):
        if (freqBand == 'delta'):
            return [None, 3.5, 2]
        elif (freqBand == 'theta'):
            return [3.5,7.5,5]
        elif (freqBand == 'alpha'):
            return [7.5,13.0,10]
        elif (freqBand == 'beta1'):
            return [13.0, 25.0, 19]
        elif (freqBand == 'beta2'):
            return [25.0, 45.0, 35]
        
    def filterFrequencyRange(self, low=None, high=None, order=8):
        print("Frequency Range")
        if (low == None and high == None):
            return
        sampleFrequency = 100        
        if (high == None): #Allow high frequency ranges
            filtb,filta = scisig.butter(order,low/(sampleFrequency/2),btype='highpass')
        elif (low == None): #allow low frequency ranges
            filtb,filta = scisig.butter(order,high/(sampleFrequency/2),btype='lowpass')
        else:        
            filtb,filta = scisig.butter(order,[low/(sampleFrequency/2),high/(sampleFrequency/2)] ,btype='bandpass')
        myError = 0
        try:
            for i in range(self.data.shape[0]):
                #print(i)
                myError = i
                self.data[i] = scisig.filtfilt(filtb,filta,self.data[i])
        except Exception as e:
            print(f"BAND RANGE ERROR epoch: {myError}")

    def fourierAll(self):
        allFreq = []
        for i in range(self.data.shape[0]):
            allFreq.append(fftpack.fft(self.data[i]))

        return allFreq
    
class keras_ann(object):
    def __init__(self):
        self.dataPath = ""
        self.outputPath = ""
        self.inputShape = (3000,1)
        self.killer = GracefulKiller()

    def updatePaths(self, dataPath = '', outputPath = ''):
        self.outputPath = outputPath
        self.dataPath = dataPath
        
    def convModel(self, modelArgs=[], inputShape = [3000,1], defaultInput=True):
        
        #
        # For First Layer, input requried
        #
        if defaultInput:
            model = Xtensor = Input(shape=inputShape)
                   
        #
        # For all other layers
        #
        #print("BUIDLING==================================")
        indexes = dict.fromkeys(['conv1d','flatten','dense','maxpool1d','avgpool1d','compile','rnn','lstm','input'],0)
        for modelArg in modelArgs:
            indexes[modelArg['layer']] += 1
            name = f"{modelArg['layer']}{indexes[modelArg['layer']]}"
            if (modelArg['layer'] == 'input'):
                model = Xtensor = Input(shape=modelArg['shape'])
            elif (modelArg['layer'] == 'conv1d'):
                #print("CONVOLUTION1D===================================================")
                model = Conv1D(filters=modelArg['no_filters'],
                                   kernel_size=modelArg['kernal_size'],
                                   padding=modelArg['padding'],
                                   activation=modelArg['activation'],
                                             name=name
                                   )(model)#shape batch, steps, channels
            elif (modelArg['layer'] == 'flatten'):
                #print("FLATTEN===================================================")
                model = Flatten()(model)

            elif (modelArg['layer'] == 'rnn'):
                if (modelArg['type'] == 'lstm'):
                    tlstm = LSTM(modelArg['units'], return_sequences = modelArg['return_sequences'])
                elif (modelArg['type'] == 'gru'):
                    tlstm = GRU(modelArg['units'], return_sequences = modelArg['return_sequences'])
                if 'wrapper' in modelArg.keys():
                    if modelArg['wrapper'] == 'timedistributed':
                        model = TimeDistributed(tlstm) (model)
                    elif modelArg['wrapper'] == 'bidirectional':
                        model = Bidirectional(tlstm)(model)
                elif 'wrappers' in modelArg.keys():
                    for wrapper in modelArg['wrappers']:
                        if wrapper == 'timedistributed':
                            tlstm = TimeDistributed(tlstm)
                        elif wrapper == 'bidirectional':
                            tlstm = Bidirectional(tlstm)
                    model = tlstm(model)
                else:
                    model = tlstm(model)

            elif (modelArg['layer'] == 'dense'):
                #print("DENSE===================================================")
                tdense = Dense(modelArg['output'],
                                  activation=modelArg['activation'],
                                  #kernel_initializer=modelArg['kernel_initializer'],
                                  #bias_initializer=modelArg['bias_initializer'],
                                             name=name
                                  )
                
                if 'wrapper' in modelArg.keys():
                    if modelArg['wrapper'] == 'timedistributed':
                        model = TimeDistributed(tdense)(model)
                else:
                    model = tdense(model)
            elif (modelArg['layer'] == 'maxpool1d'):
                #print("MAXPOOL===================================================")
                model = MaxPooling1D(pool_size=modelArg['pool_size'],
                                           strides=modelArg['strides'],
                                           padding=modelArg['padding'],
                                             name=name
                                         )(model)
            elif (modelArg['layer'] == 'avgpool1d'):
                #print("AVGPOOL===================================================")
                model = AveragePooling1D(pool_size=modelArg['pool_size'],
                                           strides=modelArg['strides'],
                                           padding=modelArg['padding'],
                                             name=name,)(model)
            elif (modelArg['layer'] == 'compile'):
                #print("COMPILECONVOLUTION1D===================================================")
                model = Model(Xtensor, model)
                self.compileModel(model,modelArg)
        return model
    def getOptimizer(self,optimizer, options):
        if (optimizer == 'sgd'):
            return optimizers.SGD(lr=options[0], momentum=options[1], nesterov=options[2])
        
        if (optimizer == 'adam'):
            return optimizers.Adam(lr=options[0], beta_1=options[1], beta_2=options[2], amsgrad=options[3])

        if (optimizer == 'nadam'):
            return optimizers.Nadam(lr=options[0], beta_1=options[1], beta_2=options[2])
        
        if (optimizer == 'rmsprop'):
            return optimizers.RMSprop(lr=options[0],rho=options[1])
        
    def compileModel(self,model,modelArg):
        #NOTE: metrics are not used for training and therefor not really needed. The loss is the important one
        if ('optimizerOptions' in modelArg.keys()):
            #print()
            model.compile(optimizer=self.getOptimizer(modelArg['optimizer'],modelArg['optimizerOptions']), #tf.train.AdamOptimizer(0.001),
                              loss=modelArg['loss'],
                              metrics = ['acc']) #tf.keras.losses.categorical_crossentropy,
        else: 
            model.compile(optimizer=modelArg['optimizer'], #tf.train.AdamOptimizer(0.001),
                              loss=modelArg['loss'],
                              metrics = ['acc']) #tf.keras.losses.categorical_crossentropy,
            

    #source: https://stackoverflow.com/questions/40496069/reset-weights-in-keras-layer
    def reset_weights(self, model):
        print("problem1")
        session = K.get_session()
        for layer in model.layers: 
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
            print("problem2")
            if hasattr(layer, 'bias_initializer'):
                layer.bias.initializer.run(session=session)
            print("problem3")

    def trainModel(self, paramSets, X, Y, valSplit=0.0, epochs=1, batchSize=None, visualize=False, saveLoc='', defaultInput=True):

        callBacks = [EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)]
        if (visualize):
            callBacks.append(TensorBoard(log_dir='./logs',
                                             histogram_freq=3,
                                             write_graph=False,
                                             write_images=False,
                                             update_freq='epoch',
                                             profile_batch=2,
                                             embeddings_freq=0,
                                             embeddings_metadata=None))
            
        modelFile = open(self.outputPath + "fileModel.csv", 'w')
        resultFile = open(self.outputPath + "fileResult.csv",'w')
        resultFile.write('modelNum|resultType|results\n')
        modelNum = 0
        for paramSet in paramSets:
            
            modelFile.write(str(modelNum) + "|")
            json.dump(paramSet, modelFile)
            modelFile.write("\n")
            print("\n\n=================\nTraining Model " + str(modelNum) + "\n=================\n")
            #print(paramSet, flush=True)
            try:
                model = self.convModel(paramSet,defaultInput=defaultInput)
                print(model.summary())
                #model.save_weights('temp_weights.h5')
                
                    
                fitHistory = model.fit(X, Y, batch_size=batchSize, verbose=0, validation_split=valSplit, epochs=epochs,callbacks=callBacks )
                
                #resultFile.write(fitHistory.history)
                #json.dump(fitHistory.history, resultFile)
                #resultFile.write("\n")
                for histItem in fitHistory.history:
                    resultFile.write(str(modelNum) + '|'
                                         + histItem + '|'
                                         + '|'.join([str(item) for item in fitHistory.history[histItem]])
                                         + '\n')
                                                      
                modelWeightFile = saveLoc + f'{modelNum}.weights.h5'
                model.save_weights(modelWeightFile)
                #model.save(modelWeightFile)
                
            except Exception as e:
                print(str(e))
                
            K.clear_session()
            modelNum+=1
            
            if self.killer.kill_now:
                print("killed")
                break
        modelFile.close()
        resultFile.close()
        
        
    
    def parameterSearch(self, paramSets, X, Y, numSplits=2,valSplit=0.0, epochs=1, batchSize=None,saveModel=False, visualize=False, saveLoc=''):
        # create CV dat LOOV 
        #numSplits = 2
        Kf = StratifiedKFold(n_splits=numSplits)
        callBacks = [EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)]
        if (visualize):
            callBacks.append(TensorBoard(log_dir='./logs',
                                             histogram_freq=3,
                                             write_graph=False,
                                             write_images=False,
                                             update_freq='epoch',
                                             profile_batch=2,
                                             embeddings_freq=0,
                                             embeddings_metadata=None))

        #for each parameter set
        # make a model
        #
        #X = [0,1,2,3,4,5,6,7,8,9]
        modelFile = open(self.outputPath + "fileModel.csv", 'w')
        resultFile = open(self.outputPath + "fileResult.csv",'w')
        resultFile.write("modelNum|True REM|False REM|False NonREM|True NonREM|Acc|Sens|Spec|Recall|Precision|f1score|finalLoss\n")
        modelNum = 0
        for paramSet in paramSets:
            
            modelFile.write(str(modelNum) + "|")
            json.dump(paramSet, modelFile)
            modelFile.write("\n")
            print("\n\n=================\nTesting Model " + str(modelNum) + "\n=================\n")
            print(paramSet, flush=True)
            try:
                model = self.convModel(paramSet)
                print(model.summary())
                model.save_weights('temp_weights.h5')
                j = 0
                for trainInd, testInd in Kf.split(X, np.argmax(Y,axis=1)):
                    #print(f"fold: {j}")
                    fitHistory = model.fit(X[trainInd], Y[trainInd], batch_size=batchSize, verbose=0, validation_split=valSplit, epochs=epochs,callbacks=callBacks )
                    if (saveModel):
                        modelWeightFile = saveLoc + f'{modelNum}.{j}.weights.h5'
                        model.save_weights(modelWeightFile)
                        #model.save(modelWeightFile)
                    Ypred = np.zeros((testInd.shape[0],Y.shape[1]))
                    Yi = 0
                    for pred in np.argmax(model.predict(X[testInd], batch_size=None), axis=1):
                        Ypred[Yi][pred] = 1
                        Yi += 1

                    
                    #NOTE:
                    #confusionMatrix = multilabel_confusion_matrix(Y[testInd], Ypred)[0]
                    ##print(confusionMatrix)
                    ##confusionMatrix = confusion_matrix(np.argmax(Y[testInd], axis=1), np.argmax(Ypred, axis=1))
                    ##print(confusionMatrix)
                    ##print('f1_score:',f1_score(Y[testInd], Ypred, average='macro'))
                    #resultFile.write(str(modelNum) + "|")
                    ##for row in confusionMatrix:
                    ##    for el in row:
                    ##        resultFile.write(str(el) + "|")
                    ##"modelNum|True REM|False NonREM|False REM|True NonREM|Acc|Sens|Spec|Recall|Precision|f1score\n"
                    #
                    #tn = confusionMatrix[0][0]
                    #fn = confusionMatrix[1][0]
                    #tp = confusionMatrix[1][1]
                    #fp = confusionMatrix[0][1]

                    tp=tn=fn=fp=0
                    Yi= 0
                    for y in Y[testInd]:
                        tp += Ypred[Yi][0]*y[0]
                        fp += max(Ypred[Yi][0]-y[0],0)
                        tn += Ypred[Yi][1]*y[1]
                        fn += max(Ypred[Yi][1]-y[1],0)
                        Yi+=1
                       
                    acc=sens=spec=prec=rec=f1=0
                    acc=(tp+tn)/(tp+tn+fp+fn)
                    if (tp+fn > 0):
                        sens=tp/(tp+fn)
                    if (tn+fp > 0):
                        spec=tn/(tn+fp)
                    if (tp+fp > 0):
                        prec=tp/(tp+fp)
                    if (tp+fn > 0):
                        rec=tp/(tp+fn)
                    if (prec+rec > 0):
                        f1=2*((prec*rec)/(prec+rec))
                    resultFile.write(f"{modelNum}|{tp:.3f}|{fp:.3f}|{fn:.3f}|{tn:.3f}|{acc:.3f}|{sens:.3f}|{spec:.3f}|{rec:.3f}|{prec:.3f}|{f1:.3f}|{fitHistory.history['loss'][-1]:10.3f}\n")
                    print(f"{'Validate':10s}|{'modelNum':10s}|{'tp':10s}|{'fp':10s}|{'fn':10s}|{'tn':10s}|{'acc':10s}|{'sens':10s}|{'spec':10s}|{'rec':10s}|{'prec':10s}|{'f1':10s}|{'loss':10s}\n")
                    print(f"{j:10d}|{modelNum:10d}|{tp:10.3f}|{fp:10.3f}|{fn:10.3f}|{tn:10.3f}|{acc:10.3f}|{sens:10.3f}|{spec:10.3f}|{rec:10.3f}|{prec:10.3f}|{f1:10.3f}|{fitHistory.history['loss'][-1]:10.3f}\n", flush=True)

                    
                    #resultFile.write(str(f1_score(Y[testInd], Ypred, average='macro')) + "|\n")
                    model.load_weights('temp_weights.h5')
                    #print("reset weights")
                    #self.reset_weights(model)
                    #print("reset weights")
                    j+=1
                
            except Exception as e:
                resultFile.write("error\n")
                print(str(e))
                
            K.clear_session()
            modelNum+=1
            
            if self.killer.kill_now:
                resultFile.write("killed\n")
                print("killed")
                break
            
        modelFile.close()
        resultFile.close()

    def testModel(self, paramSets,X,Y, weights=[], loadLoc=""):
        print("modelNum|weightSet|True REM|False REM|False NonREM|True NonREM|Acc|Sens|Spec|Recall|Precision|f1score")
        modelNum=0
        for paramSet in paramSets:
            try:
                print("loading Model")
                model = self.convModel(paramSet)
                for weightSet in weights[modelNum]:
                    print("loading: ", loadLoc + weightSet)
                    model.load_weights(loadLoc + weightSet)
                    Ypred = np.zeros((Y.shape[0],Y.shape[1]))
                    Yi = 0
                    print("predicting")
                    for pred in np.argmax(model.predict(X, batch_size=None), axis=1):
                        Ypred[Yi][pred] = 1
                        Yi += 1

                    
                    tp=tn=fn=fp=0
                    Yi= 0
                    for y in Y:
                        tp += Ypred[Yi][0]*y[0]
                        fp += max(Ypred[Yi][0]-y[0],0)
                        tn += Ypred[Yi][1]*y[1]
                        fn += max(Ypred[Yi][1]-y[1],0)
                        Yi+=1
                       
                    acc=sens=spec=prec=rec=f1=0
                    acc=(tp+tn)/(tp+tn+fp+fn)
                    if (tp+fn > 0):
                        sens=tp/(tp+fn)
                    if (tn+fp > 0):
                        spec=tn/(tn+fp)
                    if (tp+fp > 0):
                        prec=tp/(tp+fp)
                    if (tp+fn > 0):
                        rec=tp/(tp+fn)
                    if (prec+rec > 0):
                        f1=2*((prec*rec)/(prec+rec))
                    print(f"{modelNum}|{weightSet}|{tp:.3f}|{fp:.3f}|{fn:.3f}|{tn:.3f}|{acc:.3f}|{sens:.3f}|{spec:.3f}|{rec:.3f}|{prec:.3f}|{f1:.3f}")
            except Exception as e:
                print("ERROR",sys.exc_info()[0])
            modelNum += 1

    def printModel(self, paramSets, weights=[], printLoc="", loadLoc="", X=None,Y=None):
        modelNum=0
        for paramSet in paramSets:
            try:
                print("paramSet")
                model = self.convModel(paramSet)
                print(paramSet)
                for weightSet in weights[modelNum]:
                    #pass
                    print("loading: ", loadLoc + weightSet)
                    #model = load_model(loadLoc + weightSet)
                    print()
                    print("==================================")
                    print("loading: ", loadLoc + weightSet)
                    print("==================================")
                    print()
                    model.load_weights(loadLoc + weightSet)
                    model.predict(X, batch_size=None)
                    #for modelArg in paramSet[1:]:
                    #    if (modelArg['layer'] == 'conv1d'):
                    #        self.compileModel(model,modelArg)
                    to_file = printLoc+f'model.{modelNum}.{weightSet}.png'
                    print("print: ", to_file)
                    #plot_model(model, to_file=to_file)
                    #print(model.layers[0].get_config())
                    #print(model.to_json())
                    print(model.summary())
                    print("===========")
                    print("layer:",model.get_layer('conv1d1').get_weights())
                    #for layer in model.layers:
                        #print(layer.get_weights())
                    #    print(layer.get_weights())
                    return
            except Exception as e:
                print("ERROR",sys.exc_info()[0])
                print(e)
            modelNum += 1


    def getNorm(self, myDir):
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
    
    def getWeights(self,weights,myDir):

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
    
    def saveModelOutput(self, paramSets, X, Y, weights=[], saveLoc='',saveName='output.csv', loadLoc=''):
        modelNum = 0
        predictions = []
        print("modelNum|weightSet|True REM|False REM|False NonREM|True NonREM|Acc|Sens|Spec|Recall|Precision|f1score")
        for paramSet in paramSets:
            try:
                #print("loading Model", paramSet)
                model = self.convModel(paramSet)
                weightSet = weights[modelNum]
                #print("loading: ", loadLoc + weightSet)
                model.load_weights(loadLoc + weightSet)
                Ypred = np.zeros((Y.shape[0],Y.shape[1]))
                Yi = 0
                #print("predicting")
                predictions.append(model.predict(X, batch_size=None))
                for pred in np.argmax(predictions[-1], axis=1):
                    Ypred[Yi][pred] = 1
                    Yi += 1

                    
                tp=tn=fn=fp=0
                Yi= 0
                for y in Y:
                    tp += Ypred[Yi][0]*y[0]
                    fp += max(Ypred[Yi][0]-y[0],0)
                    tn += Ypred[Yi][1]*y[1]
                    fn += max(Ypred[Yi][1]-y[1],0)
                    Yi+=1
                       
                acc=sens=spec=prec=rec=f1=0
                acc=(tp+tn)/(tp+tn+fp+fn)
                if (tp+fn > 0):
                    sens=tp/(tp+fn)
                if (tn+fp > 0):
                    spec=tn/(tn+fp)
                if (tp+fp > 0):
                    prec=tp/(tp+fp)
                if (tp+fn > 0):
                    rec=tp/(tp+fn)
                if (prec+rec > 0):
                    f1=2*((prec*rec)/(prec+rec))
                print(f"{modelNum}|{weightSet}|{tp:.3f}|{fp:.3f}|{fn:.3f}|{tn:.3f}|{acc:.3f}|{sens:.3f}|{spec:.3f}|{rec:.3f}|{prec:.3f}|{f1:.3f}")
            except Exception as e:
                print("ERROR",sys.exc_info()[0])
            modelNum += 1
        #for prediction in predictions:
            
        # TO DO STACK AND PRINT PREDICTIONS
        #print(np.array(predictions).shape)
        with open(saveLoc + saveName,'w') as saveFile:
            for i in range(len(predictions[0])):
                line = str(np.argmax(Y[i])) + '|' + '|'.join([str(pred) for pred in predictions[0][i]])
                for j in range(1,5):
                    line += '|' + '|'.join([str(pred) for pred in predictions[j][i]])
                line += '\n'
                saveFile.write(line)
            #print(line)
        #allPred = np.concatenate(predictions, axis=1)
        #print(allPred.shape)
        #print(predictions[0])
            
    def buildModelStack(self, X,Y,convModel=[],auto=[],mem=[],dense=[],order=[]):
        #X = np.array(X)
        #Y = np.array(Y)
        Xtensor = Input(shape=X.shape[1:])
        print("Xtensor")
        print(X.shape)
        num_filters = 1
        convLayers = []
        #Delta Frequencies < 3Hz
        #high pass (no low pass needed)
        convLayers.append(Conv1D(filters=1, kernel_size=33, padding='same', activation='relu')(Xtensor))
        # Filter to find good wavelets
        convLayers[-1] = Conv1D(filters=num_filters, kernel_size=66, padding='same', activation='relu')(convLayers[-1])

        #Theta Frequencies 3.5-7.5 Hz
        #Low pass
        convLayers.append(Conv1D(filters=1, kernel_size=33, padding='same', activation='relu')(Xtensor))
        #High pass
        convLayers[-1] = Conv1D(filters=1, kernel_size=13, padding='same', activation='relu')(convLayers[-1])
        convLayers[-1] = Conv1D(filters=num_filters, kernel_size=20, padding='same', activation='relu')(convLayers[-1])

        #Alpha Frequencies 7.5-13 Hz
        #Low pass
        convLayers.append(Conv1D(filters=1, kernel_size=13, padding='same', activation='relu')(Xtensor))
        #High pass
        convLayers[-1] = Conv1D(filters=1, kernel_size=8, padding='same', activation='relu')(convLayers[-1])
        convLayers[-1] = Conv1D(filters=num_filters, kernel_size=10, padding='same', activation='relu')(convLayers[-1])

        #Beta(1) Frequencies 13-25 Hz
        #Low pass
        convLayers.append(Conv1D(filters=1, kernel_size=8 , padding='same', activation='relu')(Xtensor))
        #high pass
        convLayers[-1] = Conv1D(filters=1, kernel_size=4, padding='same', activation='relu')(convLayers[-1])
        convLayers[-1] = Conv1D(filters=num_filters, kernel_size=5, padding='same', activation='relu')(convLayers[-1])

        #Beta(2) Frequencies > 25 Hz
        #Low pass
        convLayers.append(Conv1D(filters=1, kernel_size=4, padding='same', activation='relu')(Xtensor))        
        convLayers[-1] = Conv1D(filters=num_filters, kernel_size=3, padding='same', activation='relu')(convLayers[-1])

        print("ConvLayers")
        
        flattenLayers = []
        flattenModels = []
        for convLayer in convLayers:
            flattenLayers.append(Flatten()(convLayer))
            flattenModels.append(Model(Xtensor,flattenLayers[-1]))
        print("flattenLayers")
            
        denseLayers = []
        for flattenLayer in flattenLayers:
            denseLayers.append(Dense(2, activation='softmax')(flattenLayer))
        print("denseLayers")

        convTrainModels = []
        for denseLayer in denseLayers:
            convTrainModels.append(Model(Xtensor,denseLayer))
            convTrainModels[-1].compile(optimizer="adam",loss="categorical_crossentropy")
            convTrainModels[-1].fit(X,Y)

        print("convTrainModels")
        #encodeLayers = []
        #for flattenLayer in flattenLayers:
        #    encodeLayers.append(Dense(3000)(flattenLayer))
        #    print(flattenLayer.shape)
        #print("ensoderLayers")
        #decodeLayers = []
        #for encodeLayer in encodeLayers:
        #    decodeLayers.append(Dense(num_filters*3000)(encodeLayer))
        #print("decodeLayers")
        #decoderModels = []
        #for decodeLayer,flattenModel in zip(decodeLayers, flattenModels):
        #    decoderModels.append(Model(Xtensor,decodeLayer))
        #    decoderModels[-1].compile(optimizer="adam",loss="mse")
        #    decoderModels[-1].fit(X, flattenModel.predict(X))
        #print("decodeModels")
        #print(convTrainModels)
        #classModel = Concatenate()([flattenLayer for flattenLayer in flattenLayers])
        
        #classModel = Embedding(
        #classModel = Dense(100, activation="relu")(classModel)
        #classModel = Dense(100, activation="relu")(classModel)
        newData = np.hstack([convTrainModel.predict(X) for convTrainModel in convTrainModels])
        Data = np.zeros(shape=(newData.shape[0]-2,10,10))
        i=0
        for n in np.arange(9,newData.shape[0],1):
            for j in np.arange(10):
                for k in np.arange(10):
                    Data[i][j][k] = newData[n-j][k]
            i+=1
                    
        #Data = [[newData[n-2],newData[n-1],newData[n]] for n in np.arange(2,newData.shape[0],1)]
        #newData = np.ndarray([convTrainModel.predict(X).flatten() for convTrainModel in convTrainModels]).flatten('F')
        
        #print(Data.shape)
        print(Data[0])
        #return
        timeTensor = Input(shape=[10,10])
        classModel = LSTM(1000)(timeTensor)
        #classModel = Flatten()(classModel)
        classModel = Dense(2, activation="softmax")(classModel)
        classModel = Model(timeTensor, classModel)
        classModel.compile(optimizer="adam",loss="categorical_crossentropy")
        classModel.fit(Data,Y[2:])
        print("ClassModel")


        Ypred = np.zeros((X.shape[0],Y.shape[1]))
        print("zeros")
        Yi = 0
        for pred in np.argmax(classModel.predict(Data, batch_size=None), axis=1):
            Ypred[Yi][pred] = 1
            Yi += 1
        print("prediction")
        tp=tn=fn=fp=0
        Yi= 0
        for y in Y[2:]:
            tp += Ypred[Yi][0]*y[0]
            fp += max(Ypred[Yi][0]-y[0],0)
            tn += Ypred[Yi][1]*y[1]
            fn += max(Ypred[Yi][1]-y[1],0)
            Yi+=1
        print("tp,tn,fp,fn")
        acc=sens=spec=prec=rec=f1=0
        acc=(tp+tn)/(tp+tn+fp+fn)
        if (tp+fn > 0):
            sens=tp/(tp+fn)
        if (tn+fp > 0):
            spec=tn/(tn+fp)
        if (tp+fp > 0):
            prec=tp/(tp+fp)
        if (tp+fn > 0):
            rec=tp/(tp+fn)
        if (prec+rec > 0):
            f1=2*((prec*rec)/(prec+rec))
        modelNum = 1
        print(f"{'modelNum':8s}|{'tp':8s}|{'fp':8s}|{'fn':8s}|{'tn':8s}|{'acc':8s}|{'sens':8s}|{'spec':8s}|{'rec':8s}|{'prec':8s}|{'f1':8s}\n")
        print(f"{modelNum:8d}|{tp:8.3f}|{fp:8.3f}|{fn:8.3f}|{tn:8.3f}|{acc:8.3f}|{sens:8.3f}|{spec:8.3f}|{rec:8.3f}|{prec:8.3f}|{f1:8.3f}\n")

            

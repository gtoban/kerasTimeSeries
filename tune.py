import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from kerasOOP import keras_ann, ann_data

def main():
    myAnn = keras_ann()
    myData = ann_data()

    data,labels,recordCount = myData.readData()
    print(data.shape)
    print(labels.shape)
    data = data[:10] #int(data.shape[0]/4)]
    labels = labels[:10] #int(labels.shape[0]/4)]
    
    model = KerasClassifier(build_fn=myAnn.convModel, epochs=2,batch_size=recordCount)

    #model.fit(data, labels, epochs=1, batch_size=recordCount)
    params = dict(no_filters=[[128,64],[64,32],[32,16]])

    random_search = RandomizedSearchCV(model, params, cv=3)
    random_search_results = random_search.fit(data,labels)
    
    print(f"Best: {random_search_results.best_score_} using {random_search_results.best_params_}")
    print(random_search_results.best_score_)
    print(random_search_results.best_params_)

main()

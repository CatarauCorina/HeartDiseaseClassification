#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:18:03 2017

@author: cataraucorina
"""

import numpy as np
import sys 
import keras
import pydotplus
import matplotlib.pyplot as plt;

sys.path.insert(0,'~/Documents/School/data-preprocessing/');
from heartPre import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Wrapper
from keras.layers import LSTM,Dropout,Flatten
from layoutga import *






def feature_scaling(x_train,x_test):
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_x=StandardScaler();
    x_train=sc_x.fit_transform(x_train)
    x_test=sc_x.transform(x_test)
    return x_train,x_test
    

def split_data(x_heart,y_heart):
    #Splitting into Trainning and test
    from sklearn.cross_validation import train_test_split
    x_train_heart,x_test_heart,y_train_heart,y_test_heart=train_test_split(x_heart,y_heart,test_size=0.10,random_state=0);    
    return x_train_heart,x_test_heart,y_train_heart,y_test_heart

def reshape_dataset(train):
    trainX = np.reshape(train, (len(train),1,train.shape[1]))
    return np.array(trainX)

def init_LSTM(trainX,trainY,init_w,activ_f,nr_in,nr_out):
   from keras.utils import plot_model
   
   
   # create and fit the LSTM network
   median=int((nr_in+nr_out)/2)
   print("init",median,init_w,activ_f,nr_out,nr_in)
   model = Sequential()
    
   model.add(LSTM(6,input_shape=(trainX.shape[1],trainX.shape[2] ), activation=activ_f))
   model.add(Dense(median,
                         init=init_w,
                         activation=activ_f))
   model.add(Dense(input_dim=median,output_dim=nr_out, activation='softmax'))
    
   plot_model_ann(model,'lstm_model_ann')
   return model


def init_ga_lstm(layout,trainX,trainY,nr_in,nr_out):

    init_w= layout.params.w_init
    activ_f= layout.params.activ_f
    activ_f_end='softmax'
    print(int(len(layout.layers)))
    classifier= Sequential()
    classifier.add(LSTM(6,input_shape=(trainX.shape[1],trainX.shape[2] ), activation=activ_f))

    
    for i in range(1,int(len(layout.layers))):
        print(layout.get_layers()[i].neurones)
        classifier.add(Dense(output_dim=int(layout.get_layers()[i].neurones),
                         input_dim=int(layout.get_layers()[i-1].neurones),
                         init=init_w,
                         activation=activ_f))

   
    #add output layer
    classifier.add(Dense(output_dim=int(layout.get_output_size()),
                         input_dim=int(layout.get_layers()[len(layout.get_layers())-1].neurones),
                         init=init_w,
                         activation=activ_f_end))
    
    plot_model_ann(classifier,'lstm_model_ga')


    return classifier  
    

    


    
def predict_kfold(ann,x_test,y_test):  
    from keras.utils.np_utils import to_categorical
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import hamming_loss

    y_pred=ann.predict(x_test)
    print("pred:",y_pred)

    y_pred=np.array(y_pred.argmax(1))
    print("test:",y_test,"pred:",y_pred)
    
   
    test_acc = np.sum(y_test == y_pred,axis=0) / x_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))
    


def kfold_simple(folds,x_train,categorical_labels,x_test,y_test):
    from sklearn.cross_validation import KFold
    layout=Layout()
    layer_1=Layer(12)
    layer_2=Layer(14)
    layer_3=Layer(37)
    layer_4=Layer(26)
    layer_5=Layer(49)
    layer_6=Layer(3)

    layers_nn=[]
    layers_nn.append(layer_1)
    layers_nn.append(layer_2)
    layers_nn.append(layer_3)
    layers_nn.append(layer_4)
    layers_nn.append(layer_5)
    layers_nn.append(layer_6)


    layout.layers=layers_nn
    param=ElementsF('elu','Nadam','glorot_normal','mean_absolute_percentage_error')
    layout.params=param
    cvscores=[]
    kfold = KFold(categorical_labels.shape[0],n_folds=folds, shuffle=True)
    for train, test in kfold:
        ann=init_ga_lstm(layout,x_train,categorical_labels,x_test.shape[1],categorical_labels.shape[1])

         #ann=init_LSTM(x_train,categorical_labels,'lecun_uniform','tanh',x_test.shape[1],categorical_labels.shape[1])
        ann.compile(layout.params.optim_f,
                layout.params.loss_f,
              metrics=['accuracy'])
        ann.fit(x_train[train],categorical_labels[train],batch_size=10,epochs=1000,verbose=0)
        scores = ann.evaluate(x_train[test],categorical_labels[test], verbose=0)
        print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    
    return cvscores
    
def ann_kfold(x_train,categorical_labels,x_test,y_test):
  print(x_test.shape[1],categorical_labels.shape[1])  
  scores=kfold_simple(5,x_train,categorical_labels,x_test,y_test)
  print("Scores:",np.mean(scores))
  return ann


def best_results_kfold_lstm(x_train,categorical_labels,x_test,y_test,outfile):
    best_res=pd.read_csv("best_results.csv",sep=',')
    i=0
    kflod_stats=pd.DataFrame(index=range(best_res.size),columns=['Activ','Init','Optim','Loss','Acc'])

    while(pd.notnull(best_res.loc[i]['Activ'])):
        print(i,".",best_res.loc[i]['Activ'])
        ann=init_LSTM(x_train,categorical_labels,best_res['Init'][i],best_res['Activ'][i],x_test.shape[1],categorical_labels.shape[1])
        ann.compile(best_res['Optim'][i],best_res['Loss'][i],metrics=['accuracy'])

        scores=kfold_simple(5,x_train,categorical_labels,ann,x_test,y_test)
        kflod_stats['Activ'][i]=best_res.loc[i]['Activ']
        kflod_stats['Init'][i]=best_res.loc[i]['Init']
        kflod_stats['Optim'][i]=best_res.loc[i]['Optim']
        kflod_stats['Loss'][i]=best_res.loc[i]['Loss']
        kflod_stats['Acc'][i]=np.mean(scores)
        i=i+1
        kflod_stats.to_csv(outfile)



def main():
    #X,Y=get_data()
    X,Y=unproc()
    fieldnames = ['BI-RADS', 'age','shape','margin','density','severity']
    mamography=data_file_to_csv('mammographic_masses.data','mamography_dataset.csv',fieldnames)
    X,Y=general_data_encode('mamography_dataset.csv')
    #split data
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
    x_train,x_test,y_train,y_test=split_data(X,Y)
    x_train,x_test=feature_scaling(x_train,x_test)
    y_test=np.array(list(y_test))
  
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(y_train, num_classes=None)
    categorical_labels_test = to_categorical(y_test, num_classes=None)

    
    # reshape input to be [samples, time steps, features]

    trainX = reshape_dataset(x_train)
    testX = reshape_dataset(x_test)
    trainY = reshape_dataset(categorical_labels)
    testY = reshape_dataset(categorical_labels_test)


    model=ann_kfold(trainX,categorical_labels,x_test,categorical_labels_test)
    best_results_kfold_lstm(trainX,categorical_labels,x_test,y_test,'kfold_lstm_mamo.csv')
    pred=model.predict(testX)
    pred=pred.argmax(1)
    test_acc = np.sum(y_test['severity'] == pred,axis=0) / x_test.shape[0]

    predict_kfold(model,testX,y_test)
    
   
   
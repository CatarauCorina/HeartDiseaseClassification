#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 01:54:42 2017

@author: cataraucorina
"""

import numpy as np
import sys 
import keras
import matplotlib.pyplot as plt;

sys.path.insert(0,'~/Documents/School/data-preprocessing/');
from heartPre import *
from keras.models import Sequential
from keras.layers import Dense
from layoutga import Layout,Layer


def classification_rate(Y,P):
    return np.mean(Y == P)

def compute_confusion_matrix(Y,T):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(Y,T)

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



def init_ann(X,Y,nr_in,optim,loss_f,init_w,activ_f,activ_f_end):
    print("Optimization:",optim,"Loss:",loss_f,"init_w:",init_w,"Acttivation:",activ_f)
    classifier= Sequential()
    classifier.add(Dense(output_dim=12,
                         input_dim=X.shape[1],
                         init=init_w,
                         activation=activ_f,
                        ))
  
    
     #second hidden layer
    classifier.add(Dense(output_dim=9,
                         input_dim=12,
                         init=init_w,
                         activation=activ_f))

   
    #add output layer
    classifier.add(Dense(output_dim=5,
                         input_dim=9,
                         init=init_w,
                         activation=activ_f_end))
    plot_model_ann(classifier,'model_ann')

    return classifier




def init_ga_ann(X,Y,nr_in,optim,loss_f,init_w,activ_f,activ_f_end):
    init_w='lecun_uniform'
    activ_f='tanh'
    activ_f_end='softmax'
    classifier= Sequential()
    classifier.add(Dense(output_dim=int(layout.get_layers()[0].neurones),
                         input_dim=int(layout.get_input_size()),
                         init=init_w,
                         activation=activ_f))
    
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
    
    plot_model_ann(classifier,'heart_simple_ga_best_'+str(len(layout.layers))+'_'+str(layout.get_layers()[0].neurones))


    return classifier

def kfold_shuffle(folds,split_size,x_train,categorical_labels,ann):
    from sklearn.cross_validation import ShuffleSplit
    #kfold = KFold(categorical_labels.shape[0],n_folds=10, shuffle=True, random_state=seed)
    ss = ShuffleSplit(categorical_labels.shape[0],n_iter=folds, test_size=split_size,random_state=0)
    for train, test in ss:
        ann.fit(x_train[train],categorical_labels[train],batch_size=10,epochs=1000,verbose=0)
        scores = ann.evaluate(x_train[test],categorical_labels[test], verbose=0)
        print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))
    return ann


def kfold_shuffle_stratified(folds,x_train,categorical_labels,ann):
    from sklearn.cross_validation import StratifiedKFold
    kfold = StratifiedKFold(categorical_labels,n_folds=folds, shuffle=True)
    for train, test in kfold:
        ann.fit(x_train[train],categorical_labels[train],batch_size=10,epochs=1000,verbose=0)
        scores = ann.evaluate(x_train[test],categorical_labels[test], verbose=0)
        print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))
    return ann


def kfold_simple(folds,x_train,categorical_labels,ann,x_test,y_test):
    from sklearn.cross_validation import KFold,cross_val_score
    
    cvscores=[]
    kfold = KFold(categorical_labels.shape[0],n_folds=folds, shuffle=True)
    for train, test in kfold:
        ann.fit(x_train[train],categorical_labels[train],batch_size=10,epochs=1000,verbose=0)
        scores = ann.evaluate(x_train[test],categorical_labels[test], verbose=0)
        print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    
    return cvscores
    
   

def predict(ann,x_test,y_test,res,count):  
    from keras.utils.np_utils import to_categorical
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import hamming_loss

    y_pred=ann.predict(x_test)
    print("pred:",y_pred)

    y_pred=np.array(y_pred.argmax(1))
    print("test:",y_test,"pred:",y_pred)
    
    compute_confusion_matrix(y_test,y_pred)
    rate=classification_rate(y_pred,y_test)
    print("Classif rate",rate)
    res['ClassifRate'][count]=rate
    
    test_acc = np.sum(y_test == y_pred,axis=0) / x_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))
    
    report = classification_report(y_test, y_pred)
    print(report)
    
    acc_score=accuracy_score(y_test, y_pred)
    print("Accuracy score",acc_score)
    res['Acc'][count]=acc_score
    
    ham=hamming_loss(y_test, y_pred)
    print("Hamming loss",ham)
    res['Hamming'][count]=ham
    
    
    
def predict_kfold(ann,x_test,y_test,res,count):  
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
    
    report = classification_report(y_test, y_pred)
    print(report)
    
    acc_score=accuracy_score(y_test, y_pred)
    print("Accuracy score",acc_score)
    res['Acc'][count]=acc_score
    
    ham=hamming_loss(y_test, y_pred)
    print("Hamming loss",ham)
    res['Hamming'][count]=ham    

    
    
            
def testing_ann(x_train,categorical_labels,x_test,y_test):
    import pandas as pd
    df = pd.read_csv("nnX.csv",sep=';')
    res = pd.read_csv("big4.csv",sep=",")
    count=0
    activations = df['Activation']
    activations=activations.dropna()
    
    init = df['Init']
    init=init.dropna()
    
    optim = df['Optim']
    optim=optim.dropna()
    
    loss=df['Loss']
    loss=loss.dropna()
    nr=0
    
    for a in range(0,activations.shape[0]):
        activ=activations[a]
        for i in range(0,init.shape[0]):
            init_w=init[i]
            for l in range(0,loss.shape[0]):
                lo=loss[l]
                for o in range(0,optim.shape[0]):
                    op=optim[o]
                    nr=nr+1
                    print(nr,".")
                    ann=init_ann(x_train,categorical_labels,13,op,lo,init_w,activ,'softmax')
                    ann.compile(optimizer=optim[o],loss=loss[l],metrics=['accuracy'])
                    ann.fit(x_train,categorical_labels,batch_size=10,epochs=500,verbose=0)
                    #ann=kfold_simple(5,x_train,categorical_labels,ann)
                    #scores = ann.evaluate(x_train, categorical_labels)
                    #print("\n%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))

                    #ann=kfold_shuffle(5,0.25,x_train,categorical_labels,ann)
                    res['Activ'][count]=activ
                    res['Init'][count]=init_w
                    res['Optim'][count]=op
                    res['Loss'][count]=lo
                    predict(ann,x_test,y_test,res,count)
                    count=count+1
                    res.to_csv('big4.csv')
                    


def best_results_kfold(x_train,categorical_labels,x_test,y_test):
    analysis= pd.read_csv("analysis.csv",sep=',')
    best_res=pd.read_csv("best_results.csv",sep=',')
    i=0
    kflod_stats=pd.DataFrame(index=range(best_res.size),columns=['Activ','Init','Optim','Loss','Acc'])

    while(pd.notnull(best_res.loc[i]['Activ'])):
         ann=init_ann(x_train,categorical_labels,13,best_res['Optim'][i],best_res['Loss'][i],best_res['Init'][i],best_res['Activ'][i],'softmax')
         ann.compile(best_res['Optim'][i],best_res['Loss'][i],metrics=['accuracy'])
         #ann.fit(x_train,categorical_labels,batch_size=10,epochs=500,verbose=0)
         scores=kfold_simple(5,x_train,categorical_labels,ann,x_test,y_test)
         #ann=kfold_shuffle(5,0.25,x_train,categorical_labels,ann)
         kflod_stats['Activ'][i]=best_res.loc[i]['Activ']
         kflod_stats['Init'][i]=best_res.loc[i]['Init']
         kflod_stats['Optim'][i]=best_res.loc[i]['Optim']
         kflod_stats['Loss'][i]=best_res.loc[i]['Loss']
         kflod_stats['Acc'][i]=np.mean(scores)
         i=i+1
         kflod_stats.to_csv('kfold_results.csv')



def main():
    #X,Y=get_data()
    X,Y=unproc()
    #split data
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
    x_train,x_test,y_train,y_test=split_data(X,Y)
    x_train,x_test=feature_scaling(x_train,x_test)
    y_test=np.array(list(y_test))
  
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(y_train, num_classes=None)
    testing_ann(x_train,categorical_labels,x_test,y_test)
    best_results_kfold(x_train,categorical_labels,x_test,y_test)

  

if __name__== '__main__':
    main()
    
    



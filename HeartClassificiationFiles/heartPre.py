#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:50:07 2017

@author: cataraucorina
"""

#Data preprocessing heart-dataset

import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
import pydotplus as pydot
import urllib.request;
import hashlib as hb;
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Wrapper
os.environ['R_HOME']='/Library/Frameworks/R.framework/Resources'

#dataset = urllib.request.urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", "processed.clevelandpy.data");
#m = hb.md5();

import rpy2.robjects as robjects
import pandas.rpy.common as com

from rpy2 import * 

def transform_to_csv(heart):
    import csv

    with open('heart-neural.csv', 'w') as csvfile:
        fieldnames = ['age', 'gender','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in heart.iterrows():
            writer.writerow({'age':row[1]['age'],
                         'gender':row[1]['sex'],
                         'cp':row[1]['cp'],
                         'trestbps':row[1]['trestbps'],
                         'chol':row[1]['chol'],
                         'fbs':row[1]['fbs'],
                         'restecg':row[1]['restecg'],
                         'thalach':row[1]['thalach'],
                         'exang':row[1]['exang'],
                         'oldpeak':row[1]['oldpeak'],
                         'slope':row[1]['slope'],
                         'ca':row[1]['ca'],
                         'thal':row[1]['thal'],
                         'num':row[1]['num']})


  
def data_file_to_csv(file,output_file,fieldnames):
    import csv

    with open(file) as input_file:
        lines = input_file.readlines()
        newLines = []
        for line in lines:
            line_new=[x.strip() for x in line.split(',')]
            newLines.append(line_new)
    
    with open(output_file, 'w') as test_file:
        writer = csv.DictWriter(test_file, fieldnames=fieldnames)
        writer.writeheader()
        file_writer = csv.writer(test_file)
        file_writer.writerows( newLines )
    
def general_data_encode(data):
    #take care of missing data
    my_data = pd.read_csv(data,sep=",")
    my_data=my_data.replace(['?'],[np.NAN])
    my_data=my_data.fillna(method='ffill')
    x_data=my_data.iloc[:,:-1]
    y_data=my_data.iloc[:,-1:]
    return x_data,y_data
    


def plot_model_ann(model,to_file,show_shapes=True, show_layer_names=True):

    dot = pydot.Dot()
    dot.set('rankdir', 'BT')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # Create graph nodes.
    for layer in layers:
        layer_id = str(id(layer))

         # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)
   

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)


        # Connect nodes with edges.
        for layer in layers:
            layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    #dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))

    dot.write(to_file, format='jpg')



def unsplit_data():
    robj=robjects.r.load('heart-proc.RData')        
    
    heart=com.load_data(robj[2])

    heart=heart.fillna(method='ffill')
    #Take care of missing data
    heart.iloc[:,11]=heart.iloc[:,11].fillna(method='ffill')
    heart.iloc[:,12]=heart.iloc[:,12].fillna(method='ffill')


    #Encoding categorical data
    from sklearn.preprocessing import LabelEncoder
    label_X_gender= LabelEncoder()
    heart[['sex']]=label_X_gender.fit_transform(heart[['sex']])

    label_X_fbs= LabelEncoder()
    heart[['fbs']]=label_X_fbs.fit_transform(heart[['fbs']])

    label_X_exang= LabelEncoder()
    heart[['exang']]=label_X_exang.fit_transform(heart[['exang']])


    label_X_cp= LabelEncoder()
    heart[['cp']]=label_X_cp.fit_transform(heart[['cp']])
    label_X_restecg= LabelEncoder()
    heart[['restecg']]=label_X_restecg.fit_transform(heart[['restecg']])
    label_X_slope= LabelEncoder()
    heart[['slope']]=label_X_slope.fit_transform(heart[['slope']])
    label_X_thal= LabelEncoder()
    heart[['thal']]=label_X_thal.fit_transform(heart[['thal']])

    return heart

def encode_data():
    robj=robjects.r.load('heart-proc.RData')        
    
    heart=com.load_data(robj[2])

    heart=heart.fillna(method='ffill')
    #Take care of missing data
    heart.iloc[:,11]=heart.iloc[:,11].fillna(method='ffill')
    heart.iloc[:,12]=heart.iloc[:,12].fillna(method='ffill')


    #Encoding categorical data
    from sklearn.preprocessing import LabelEncoder
    label_X_gender= LabelEncoder()
    heart[['sex']]=label_X_gender.fit_transform(heart[['sex']])

    label_X_fbs= LabelEncoder()
    heart[['fbs']]=label_X_fbs.fit_transform(heart[['fbs']])

    label_X_exang= LabelEncoder()
    heart[['exang']]=label_X_exang.fit_transform(heart[['exang']])


    label_X_cp= LabelEncoder()
    heart[['cp']]=label_X_cp.fit_transform(heart[['cp']])
    label_X_restecg= LabelEncoder()
    heart[['restecg']]=label_X_restecg.fit_transform(heart[['restecg']])
    label_X_slope= LabelEncoder()
    heart[['slope']]=label_X_slope.fit_transform(heart[['slope']])
    label_X_thal= LabelEncoder()
    heart[['thal']]=label_X_thal.fit_transform(heart[['thal']])
    transform_to_csv(heart)

    #x features vector
    x_heart=heart.iloc[:,0:13].values
    y_heart=heart.iloc[:,13].values
    return x_heart,y_heart
    

def get_data():
    x_heart,y_heart=encode_data()
    #Encoding categorical data
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder_cp=OneHotEncoder(categorical_features=[2])
    x_heart=onehotencoder_cp.fit_transform(x_heart).toarray()
 
    onehotencoder_restecg=OneHotEncoder(categorical_features=[9])
    x_heart=onehotencoder_restecg.fit_transform(x_heart).toarray()

    onehotencoder_slope=OneHotEncoder(categorical_features=[15])
    x_heart=onehotencoder_slope.fit_transform(x_heart).toarray()

    onehotencoder_thal=OneHotEncoder(categorical_features=[19])
    x_heart=onehotencoder_thal.fit_transform(x_heart).toarray()
    
    return x_heart,y_heart

def unproc():
    x_heart_un,y_heart_un=encode_data()
    return x_heart_un,y_heart_un







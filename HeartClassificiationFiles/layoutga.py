#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 00:42:33 2017

@author: cataraucorina
"""


import numpy as np

class Layout(object):

    def __init__(self,
                 input_size=13,
                 output_size=5,
                 nr_layers=1):
        print(self,input_size,output_size)
        self.input_size =input_size
        self.output_size = output_size
        self.accuracy=0.
        self.layers=list()
        self.params= np.nan 
      
    
    def train_nn(self,ann,x_test,y_test):
        from sklearn.metrics import accuracy_score

        y_pred=ann.predict(x_test)
        print("pred:",y_pred)

        y_pred=np.array(y_pred.argmax(1))
        print("test:",y_test,"pred:",y_pred)

        acc_score=accuracy_score(y_test, y_pred)
        return acc_score
    
 
            
        
    def get_input_size(self):
        return self.input_size
    
    def get_output_size(self):
        return self.output_size
    
    def get_layers(self):
        return self.layers
    
  
class ElementsF(object):
      def __init__(self,
                 activ_f,
                 optim_f,
                 w_init,
                 loss_f):
    
        self.activ_f=activ_f
        self.optim_f=optim_f
        self.w_init=w_init
        self.loss_f=loss_f


class Layer(object):

    def __init__(self, nr_neurones):
        self.neurones=nr_neurones
        
  
    
    def get_nr_neurones(self):
        return self.neurones
       



class Objective(object):

    def __init__(self, objective):
        self.objective = objective

    def todict(self):
        return dict(vars(self))



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 00:01:39 2017

@author: cataraucorina
"""

import json
import logging
import numpy as np
from os import path
import random
from keras import backend as K
import tensorflow as tf
from layoutga import *
from keras.layers import Dense,Dropout
from keras.models import Sequential
from deap import creator, base, tools
from deap.tools import History
from heartPre import *
from elmFeature import *
from deap import gp
from sklearn.decomposition import PCA, KernelPCA
from keras.callbacks import TensorBoard
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools 


toolbox = base.Toolbox()

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_initial_pop():
    with open('init_pop_ga_28_june.txt') as openfileobject:
        for line in openfileobject:
            l=line.split()
            param=ElementsF("","","","")
            if len(l)>1:
                if RepresentsInt(l[0])== False:
                    values=[]
                    layout=Layout()
                    for i in range(0,len(l)):
                        values.append(l[i])
                    
                    param=ElementsF(values[0],values[1],values[3],values[2])
                else:
                    nr_layers=int(l[0])
                    layers_nn=[]
                    for j in range(1,nr_layers+1):
                        layer=Layer(l[j])
                        layers_nn.append(layer)
                    layout.layers=layers_nn
           

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')





def split_data(x_heart,y_heart):
    #Splitting into Trainning and test
    from sklearn.cross_validation import train_test_split
    x_train_heart,x_test_heart,y_train_heart,y_test_heart=train_test_split(x_heart,y_heart,test_size=0.10,random_state=0);    
    return x_train_heart,x_test_heart,y_train_heart,y_test_heart

def feature_scaling(x_train,x_test):
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_x=StandardScaler();
    x_train=sc_x.fit_transform(x_train)
    x_test=sc_x.transform(x_test)
    return x_train,x_test

def create_random_nn(layout,individual_type):
    print(individual_type)
    activations=['relu', 'elu', 'tanh', 'sigmoid','softsign','softplus']
    optimizer=['RMSprop', 'Adam', 'SGD', 'Adagrad',
                      'Adadelta', 'Adamax', 'Nadam']
    loss=['mean_squared_error',
          'mean_absolute_error',
          'mean_absolute_percentage_error',
          'mean_squared_logarithmic_error',
          'squared_hinge',
          'hinge',
          'categorical_crossentropy',
          'kullback_leibler_divergence',
          'poisson',
          'cosine_proximity']
    weight_init=['lecun_uniform',
                 'glorot_normal',
                 'he_normal',
                 'he_uniform']
    layersNN=list()
    nrlayers=random.randint(1,10)
    for i in range(0,nrlayers):
        layersNN.append(Layer(random.randint(2,50)))
    mylayout=Layout()
    mylayout.input_size=13
    mylayout.output_size=5
    mylayout.nr_layers=nrlayers
    
    mylayout.layers=layersNN
    ind=individual_type(mylayout)
    ind.input_size=13
    ind.output_size=5
    ind.layers=layersNN
    ind.nr_layers=nrlayers
    activ_f=random.choice(activations)
    optim_f=random.choice(optimizer)
    loss_f=random.choice(loss)
    weight_f=random.choice(weight_init)
    param=ElementsF(activ_f,optim_f,weight_f,loss_f)
    ind.params=param
    return ind


def my_init(shape, dtype=None):
    print(np.empty(shape))
   
    return K.random_normal(shape,dtype=dtype)
    

def evaluate_nn(layout):
    X,Y=unproc()
    #split data
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
    x_train,x_test,y_train,y_test=split_data(X,Y)
    x_train,x_test=feature_scaling(x_train,x_test)
    y_test=np.array(list(y_test))
  
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(y_train, num_classes=None)
    categorical_labels_test = to_categorical(y_test, num_classes=None)

    ann=init_ann_model(layout)
    ann.compile(layout.params.optim_f,
                layout.params.loss_f,
                metrics=['accuracy'])
    ann.fit(x_train,categorical_labels,batch_size=10,epochs=500,verbose=0)
   
    score = ann.evaluate(x_test, categorical_labels_test, verbose=0)

    return score[1],

def fit_invalid_individuals(population):
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if len(invalid_ind) == 0:
        return
    fitnesses = toolbox.evaluate(invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        if fit is None:
            fit = 0
        ind.fitness.values = fit
        
        

def mutate(network):
    activations=['relu', 'elu', 'tanh', 'sigmoid','softsign','softplus']
    optimizer=['RMSprop', 'Adam', 'SGD', 'Adagrad',
                      'Adadelta', 'Adamax', 'Nadam']
    loss=['mean_squared_error',
          'mean_absolute_error',
          'mean_absolute_percentage_error',
          'mean_squared_logarithmic_error',
          'squared_hinge',
          'hinge',
          'categorical_crossentropy',
          'kullback_leibler_divergence',
          'poisson',
          'cosine_proximity']
    weight_init=['lecun_uniform',
                 'glorot_normal',
                 'he_normal',
                 'he_uniform']
    mutate_answear=[0,1]
    layersNN=network.layers
    newLayers=[]
    #mutate activ?
    if(random.choice(mutate_answear)==1):
        network.params.activ_f=random.choice(activations)
     #mutate optimizer?
    if(random.choice(mutate_answear)==1):
        network.params.optim_f=random.choice(optimizer)
     #mutate loss?
    if(random.choice(mutate_answear)==1):
        network.params.loss_f=random.choice(loss)
     #mutate weight init?
    if(random.choice(mutate_answear)==1):
        network.params.w_init=random.choice(weight_init)
        
    
    #mutate nr of layers?
    if(random.choice(mutate_answear)==1):
        nrlayers=random.randint(1,20)
        network.nr_layers=nrlayers
        for i in range(0,network.nr_layers):
            newLayers.append(Layer(random.randint(2,50))) 
      
        
        if(network.nr_layers > len(layersNN)):
            for nn in (0,len(layersNN)-1):
                #mutate neurones?
                if(random.choice(mutate_answear)==0):
                    newLayers[nn].neurones=layersNN[nn].neurones
        else:
            for nl in (0,network.nr_layers-1):
                if(random.choice(mutate_answear)==0):
                    newLayers[nl].neurones=layersNN[nl].neurones
        network.layers=newLayers
    else:
        for i  in range(0,network.nr_layers):
            #mutate nr of neurones for this layer?
            if(random.choice(mutate_answear)==1):
                layersNN[i]=Layer(random.randint(2,50))
        
   
    return network



        
def mate(mother, father):
    children = []
    layersNN=[]
    neurones_possibilities=[]
    layers_possibilities=[]
    activ_poss=[]
    optim_poss=[]
    loss_poss=[]
    w_poss=[]
    
    activ_poss.append(mother.params.activ_f)
    optim_poss.append(mother.params.optim_f)
    loss_poss.append(mother.params.loss_f)
    w_poss.append(mother.params.w_init)
    
    activ_poss.append(father.params.activ_f)
    optim_poss.append(father.params.optim_f)
    loss_poss.append(father.params.loss_f)
    w_poss.append(father.params.w_init)
    
    layers_possibilities.append(mother.nr_layers)
    layers_possibilities.append(father.nr_layers)
    for m in range(0,mother.nr_layers):
        neurones_possibilities.append(mother.layers[m].neurones)
    for f in range(0,father.nr_layers):
        neurones_possibilities.append(father.layers[f].neurones)
        
    for i in range(2):
        offspring={}
        offspring=create_random_nn(offspring,creator.Individual)
        offspring.nr_layers=random.choice(layers_possibilities)
        for j in range(0,offspring.nr_layers):
            nr_neurones=random.choice(neurones_possibilities)
            layersNN.append(Layer(nr_neurones))
        offspring.layers=layersNN
       
        activ=random.choice(activ_poss)
        optim=random.choice(optim_poss)
        loss=random.choice(loss_poss)
        weight=random.choice(w_poss)
        param=ElementsF(activ,optim,weight,loss)
        offspring.params=param
        children.append(offspring)

    return children
    


def init_ga():
    layout=Layout()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create(
        "Individual",
        Layout,
        fitness = creator.FitnessMax)  # @UndefinedVariable
    toolbox.register(
        "individual",
        create_random_nn,
        layout,
       creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selRoulette)
    toolbox.register("evaluate",ann_kfold)
   
    return toolbox
    
    
def init_ann_model(layout):
    init_w= layout.params.w_init
    activ_f= layout.params.activ_f
    activ_f_end='softmax'
    print(int(len(layout.layers)))
    classifier= Sequential()
    classifier.add(Dense(output_dim=int(layout.get_layers()[0].neurones),
                         input_dim=int(layout.get_input_size()),
                         kernel_initializer=init_w,
                         activation=activ_f))
    classifier.add(Dropout(0.5))
    for i in range(1,int(len(layout.layers))):
        print(layout.get_layers()[i].neurones)
        classifier.add(Dense(output_dim=int(layout.get_layers()[i].neurones),
                         input_dim=int(layout.get_layers()[i-1].neurones),
                         kernel_initializer=init_w,
                         activation=activ_f))
        classifier.add(Dropout(0.5))

   
    #add output layer
    classifier.add(Dense(output_dim=int(layout.get_output_size()),
                         input_dim=int(layout.get_layers()[len(layout.get_layers())-1].neurones),
                         kernel_initializer=init_w,
                         activation=activ_f_end))
    
    plot_model_ann(classifier,'ga_'+str(len(layout.layers))+'_'+str(layout.get_layers()[0].neurones))


    return classifier  


def ann_kfold(layout):
    print(layout.params.loss_f," ",layout.params.optim_f," ",layout.params.activ_f)
   
    X,Y=unproc()
    #split data
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
    x_train,x_test,y_train,y_test=split_data(X,Y)
    x_train,x_test=feature_scaling(x_train,x_test)
    y_test=np.array(list(y_test))
  
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(y_train, num_classes=None)
    categorical_labels_test = to_categorical(y_test, num_classes=None)


    scores=kfold_simple(3,x_train,categorical_labels,x_test,y_test,layout)
    print("Scores:",np.mean(scores))

   
    return np.mean(scores),


def ann_kfold_feature_scaling(layout,func):
    print(layout.params.loss_f," ",layout.params.optim_f," ",layout.params.activ_f)
    X,Y=unproc()
    scores=kfold_simple_feature(5,X,Y,layout,func)
    print("Scores:",np.mean(scores))
    
    return np.mean(scores),

def no_kfold(layout):
    from sklearn.cross_validation import KFold,cross_val_score
    cbk=TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True,write_grads=True, write_images=True)
    class_name=['0','1','2','3','4']
    cvscores=[]
    print(layout.params.loss_f," ",layout.params.optim_f," ",layout.params.activ_f)
   

    X,Y=unproc()
    #split data
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
    x_train,x_test,y_train,y_test=split_data(X,Y)
    #x_train,x_test=feature_scaling(x_train,x_test)
    #y_test=np.array(list(y_test))
  
    from keras.utils.np_utils import to_categorical
    #categorical_labels = to_categorical(y_train, num_classes=None)
    #categorical_labels_test = to_categorical(y_test, num_classes=None)

    new_xtrain,new_ytrain,new_testx,new_testy=get_lasso_lars_features(x_train,y_train,
                                       x_test,y_test)
    y_test=np.array(list(new_testy))
    layout.input_size=new_testx.shape[1]
    ann=init_ann_model(layout)
    ann.compile(layout.params.optim_f,
                layout.params.loss_f,
              metrics=['accuracy'])
    
    ann.fit(new_xtrain,new_ytrain,batch_size=10,epochs=10000,verbose=1,callbacks=[cbk])
    categorical_labels_test = to_categorical(y_test, num_classes=None)
    scores = ann.evaluate(new_testx,categorical_labels_test, verbose=0)
    y_pred=ann.predict(new_testx)
    y_pred=np.array(y_pred.argmax(1))
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_name,
                      title='Confusion matrix')
    print("Scores:",np.mean(scores))
    
    return np.mean(scores),
    



def kfold_simple_feature(folds,x_train,categorical_labels,layout,func):
    from sklearn.cross_validation import KFold,StratifiedKFold,cross_val_score
    cbk=TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
    print(func.__name__)
    cvscores=[]
    kfold = KFold(categorical_labels.shape[0],n_folds=folds, shuffle=True)
    for train, test in kfold:
      
        new_x,new_y,test_x,test_y=func(x_train[train],categorical_labels[train],
                                       x_train[test],categorical_labels[test])
        
       
        layout.input_size=new_x.shape[1]
        ann=init_ann_model(layout)
        ann.compile(layout.params.optim_f,
                layout.params.loss_f,
              metrics=['accuracy'])
        
        ann.fit(new_x,new_y,batch_size=10,epochs=1000,verbose=0, callbacks=[cbk])        
        scores = ann.evaluate(test_x,test_y, verbose=0)
        print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))
        
        cvscores.append(scores[1] * 100)
    
    return cvscores





def kfold_simple(folds,x_train,categorical_labels,testx,testy,layout):
    from sklearn.cross_validation import KFold,cross_val_score
    cbk=TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
    class_name=['0','1','2','3','4']
    cvscores=[]
    kfold = KFold(categorical_labels.shape[0],n_folds=folds, shuffle=True)
    for train, test in kfold:
        ann=init_ann_model(layout)
        ann.compile(layout.params.optim_f,
                layout2.params.loss_f,
              metrics=['accuracy'])
    
        ann.fit(x_train[train],categorical_labels[train],batch_size=10,epochs=1000,verbose=0, callbacks=[cbk])
        scores = ann.evaluate(x_train[test],categorical_labels[test], verbose=0)
      
        print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))
        y_pred=ann.predict(testx)
        y_pred=np.array(y_pred.argmax(1))
        cnf_matrix = confusion_matrix(testy, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_name,
                      title='Confusion matrix')
        cvscores.append(scores[1] * 100)
    
    return cvscores
    




def plot_result_population(pop):
    for i in range(0,len(pop)):
        res = init_ann_model(pop[i])
        plot_model_ann(res,'res_of_ga_'+str(i))
        

def write_population_to_file(pop,name):
    file = open(str(name)+".txt","w") 
    for i in range(0,len(pop)):
        file.write("Population_"+str(i)+"\n") 
        file.write(" Functions: Activ "+str(pop[i].params.activ_f)+
                   " Optim "+str(pop[i].params.optim_f)+
                   " Loss "+str(pop[i].params.loss_f)+
                   " Weight init "+str(pop[i].params.w_init)+"\n")
        file.write("Nr_layers :"+str(len(pop[i].get_layers()))+"\n") 
        for j in range(0,len(pop[i].get_layers())):
            file.write("Neurones for layer_"+str(j)+" "+str(pop[i].get_layers()[j].neurones)+"\n")
            file.write("--------")
        file.write("Fitness:"+str(pop[i].fitness.values))
        file.write("---------------------------------------------------------------") 
 
    file.close()
    



def min_fitness(pop):
    min_fit=100
    min_ind=0
    for i in range(0,len(pop)):
        if(min(min_fit,pop[i].fitness.values[0]) < min_fit):
            min_fit=min(min_fit,pop[i].fitness.values[0])
            min_ind=pop[i]
    return min_ind

def avg_fitness(pop):
    avg=0
    for i in range(0,len(pop)-1):
        avg=avg+pop[i].fitness.values[0]
    return avg/len(pop)

def plague(pop):
    doomed_ind=avg_fitness(pop)
    good_pop=[]
    for i in range(0,len(pop)-1):
        print(pop[i].fitness.values[0])
        if(float(pop[i].fitness.values[0]) > float(doomed_ind)) and float(pop[i].fitness.values[0]) >=50 :
           good_pop.append(pop[i])
   
    return good_pop
        
        
    
def print_fitness(pop):
    print(avg_fitness(pop))
    for i in range(0,len(pop)):
        print(pop[i].fitness.values[0])
        

def accuracy_decrease_stopping_condition():
   
    return AccuracyDecreaseStoppingCondition(
        min_epoch=2,
        max_epoch=10,
        noprogress_count=5)



def draw_genealogy_plot():
    import matplotlib.pyplot as plt
    import networkx

    graph = networkx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()     # Make the grah top-down
    colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    networkx.draw(graph, node_color=colors)
    plt.show()



def init_pop():
    pop = toolbox.population(n=50)

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    write_population_to_file(pop,"50_pop_initial")

    return pop


def init_pop_with_cond():
    pop_cond=toolbox.population(n=0)
    
    nr_ind=0
    while (nr_ind < 10):
        ind=toolbox.individual()
        evaluation=toolbox.evaluate(ind)
        if(evaluation[0] > 56):
            ind.fitness.values=evaluation
            pop_cond.append(ind)
            nr_ind=nr_ind+1
    write_population_to_file(pop_cond,"good_pop_1_july")

    return pop_cond
        

def evolve_2(pop):
    CXPB, MUTPB, NGEN = 0.5, 0.3, 10

    write_population_to_file(pop,"init_pop_ga")
    for g in range(NGEN):
        print('-----------Evolving generation %d' % g ,'---- nr: %d' % len(pop))
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop=plague(pop)
        print('--Population for gen: %d' % g,'-after plague: %d' % len(pop))
        pop[:] = offspring
        
    plot_result_population(pop)
    write_population_to_file(pop,"ga_pop_")
    best_individual=tools.selBest(pop,1)
    best=[]
    best.append(best_individual)
    #write_population_to_file(best,"best_ind_ga_")
    #draw_genealogy_plot()
    return best




def evaluate_feature_selection_methods():
    
    file = open("Feature_selection.txt","w") 
    layout2=Layout()

    layer_12=Layer(25)
    layer_22=Layer(35)
    layer_32=Layer(39)
    layer_42=Layer(43)
    layer_52=Layer(6)
   

    layers_nn2=[]
    layers_nn2.append(layer_12)
    layers_nn2.append(layer_22)
    layers_nn2.append(layer_32)
    layers_nn2.append(layer_42)
    layers_nn2.append(layer_52)

    layout2.layers=layers_nn2
    param2=ElementsF('sigmoid','Adagrad','he_normal','squared_hinge')
    layout2.params=param2
    
    
    no_kfold(layout2)
    
    layout=Layout()
    layer_1=Layer(7)
   

    layers_nn=[]
    layers_nn.append(layer_1)

    layout.layers=layers_nn
    param=ElementsF('tanh','Nadam','lecun_uniform','kullback_leibler_divergence')
    layout.params=param


    
    
    myFuncs=[get_lasso_lars_features]
    for i in range(0,len(myFuncs)):
        file.write("Feature selection method : "+str(myFuncs[i].__name__)+"\n")
        res=ann_kfold_feature_scaling(layout2,myFuncs[i])
        file.write("Score: "+str(res[0])+"\n")
        file.write("---------------------------------------------------------------") 

    file.close()


        



def main():
    X,Y=unproc()
    #split data
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
    x_train,x_test,y_train,y_test=split_data(X,Y)
    x_train,x_test=feature_scaling(x_train,x_test)
    y_test=np.array(list(y_test))
    
    
  
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(y_train, num_classes=None)
    t=init_ga()
    pop=init_pop()
    best=evolve_2(pop)
    print(best)
    evolve(population=toolbox.population())
    
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
    
    
    
    layout2=Layout()

    layer_12=Layer(25)
    layer_22=Layer(35)
    layer_32=Layer(39)
    layer_42=Layer(43)
    layer_52=Layer(6)
   

    layers_nn2=[]
    layers_nn2.append(layer_12)
    layers_nn2.append(layer_22)
    layers_nn2.append(layer_32)
    layers_nn2.append(layer_42)
    layers_nn2.append(layer_52)


    layout2.layers=layers_nn2
    param2=ElementsF('sigmoid','Adagrad','he_normal','squared_hinge')
    layout2.params=param2
    ann_kfold(layout2)
    no_kfold(layout2)
    
    
    
    layout3=Layout()

    layer_13=Layer(4)
    layer_23=Layer(18)
    layer_33=Layer(6)
    layer_43=Layer(31)
   

    layers_nn3=[]
    layers_nn3.append(layer_13)
    layers_nn3.append(layer_23)
    layers_nn3.append(layer_33)
    layers_nn3.append(layer_43)


    layout3.layers=layers_nn3
    param3=ElementsF('softsign','Adadelta','lecun_uniform','poisson')
    layout3.params=param3
    ann_kfold(layout3)
    no_kfold(layout3)
    
    
    
    
    
    layout2=Layout()
    layer_12=Layer(25)
    layer_22=Layer(35)
    layer_32=Layer(39)
    layer_42=Layer(43)
    layer_52=Layer(6)
   

    layers_nn2=[]
    layers_nn2.append(layer_12)
    layers_nn2.append(layer_22)
    layers_nn2.append(layer_32)
    layers_nn2.append(layer_42)
    layers_nn2.append(layer_52)


    layout2.layers=layers_nn2
    param2=ElementsF('tanh','Nadam','lecun_uniform','kullback_leibler_divergence')
    layout2.params=param2




    layout3=Layout()
    layer_31=Layer(33)
    layer_32=Layer(37)
    layer_33=Layer(13)
    layer_34=Layer(13)
    layer_35=Layer(24)
    layer_36=Layer(22)
    layer_37=Layer(20)
    layer_38=Layer(22)
    layer_39=Layer(14)


    layers_nn3=[]
    layers_nn3.append(layer_31)
    layers_nn3.append(layer_32)
    layers_nn3.append(layer_33)
    layers_nn3.append(layer_34)
    layers_nn3.append(layer_35)
    layers_nn3.append(layer_36)
    layers_nn3.append(layer_37)
    layers_nn3.append(layer_38)
    layers_nn3.append(layer_39)

 
    layout3.layers=layers_nn3
    param3=ElementsF('softplus','SGD','he_normal','squared_hinge')
    layout3.params=param3

    
    ann_kfold(layout2)
    
    evaluate_feature_selection_methods()
     
    
    
    
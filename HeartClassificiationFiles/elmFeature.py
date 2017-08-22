#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:31:15 2017

@author: cataraucorina
"""

from heartPre import *
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier



def feature_scaling_reduction(X):
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_x=StandardScaler();
    X=sc_x.fit_transform(X)
  
    return X

def get_trim_test_data(X,values):
    return X[:,values]
        

def unprocessed_features():
    X,Y=unproc()
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    return X,categorical_labels


def univ_selection(X,Y):
    #Perform feature extraction of the heart_dataset
    #Statistical test: chi^2 for non-negative features
    selected_data=SelectKBest(score_func=chi2,k=7)
    fit=selected_data.fit(X,Y)
    np.set_printoptions(precision=3)
    #for i in range(0,len(fit.scores_)):
     #   print(fit.scores_[i])
    new_features=fit.transform(X)
    return new_features


def get_univ_selection(X,Y):
    x_new=univ_selection(X,Y)
    x_scaled=feature_scaling_reduction(x_new)
    
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    return x_scaled,categorical_labels


def linear_svc(X,Y):
   lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
   model = SelectFromModel(lsvc, prefit=True)
   X_new_linear_svc = model.transform(X)  
   values= SelectFromModel.get_support(model,indices=True)
   return X_new_linear_svc,values

    
def get_linear_svc_data(X,Y,testX,testY):
    x_scaled=feature_scaling_reduction(X)
    testX_scaled=feature_scaling_reduction(testX)
    x_new,values=linear_svc(x_scaled,Y)
    testX_reduced=get_trim_test_data(testX,values)
    
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    
    
    for i in range(len(testY)):
        testY[i]=int(testY[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels_test = to_categorical(testY, num_classes=None)
    
    return x_new,categorical_labels,testX_reduced,categorical_labels_test


def tree_based(X,Y):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    values= SelectFromModel.get_support(model,indices=True)
    X_new = model.transform(X)
   
    return X_new,values 

def get_tree_data(X,Y,testX,testY):
    x_scaled=feature_scaling_reduction(X)
    test_scaled=feature_scaling_reduction(testX)
    x_new,values=tree_based(x_scaled,Y)
    testX_reduced=get_trim_test_data(test_scaled,values)

    
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    
    
    for i in range(len(testY)):
        testY[i]=int(testY[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels_test = to_categorical(testY, num_classes=None)
    return x_new,categorical_labels,testX_reduced,categorical_labels_test


def recursive_feature_selection(model,X,Y):
    rfe = RFE(model, 3)
    fit = rfe.fit(X, Y)
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_
    new_features=fit.transform(X)
    
    
    return new_featuresa

def get_recc_features(model):
    X,Y=unproc()
    x_scaled=feature_scaling_reduction(X)
    x_new=recursive_feature_selection(model,x_scaled,Y)
   
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    return x_new,categorical_labels
    
def pca_selection(X,Y):
    pca=PCA()
    fit=pca.fit(X)
    new_features=fit.transform(X)
    return new_features

def get_pca_features(X,Y):
    x_new=pca_selection(X,Y)
    x_scaled=feature_scaling_reduction(x_new)

    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    return x_scaled,categorical_labels

def extra_tree(X,Y):
    #selection using ExtraTreeClassif
    my_model=ExtraTreesClassifier()
    fit=my_model.fit(X,Y)
    sfm = SelectFromModel(fit,prefit=True)
   
    values= SelectFromModel.get_support(sfm,indices=True)
    for i in range(0,len(values)):
        print(values[i])
    new_features=fit.transform(X)
    return new_features,values

def get_extra_tree_features(X,Y,testX,testY):
    x_new,values=extra_tree(X,Y)
    x_scaled=feature_scaling_reduction(x_new)
    testX_scaled=feature_scaling_reduction(testX)
    testX_reduced=get_trim_test_data(testX_scaled,values)


    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    
    for i in range(len(testY)):
        testY[i]=int(testY[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels_test = to_categorical(testY, num_classes=None)
    return x_scaled,categorical_labels,testX_reduced,categorical_labels_test
    

def elastic_net(X,Y):
    print(X.shape)
    clf = MultiTaskElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, 
                                normalize=False, max_iter=1000, tol=0.0001, cv=None, copy_X=True, 
                                verbose=0, n_jobs=1, random_state=None, selection='cyclic')
    
    fit=clf.fit(X,Y)
    sfm = SelectFromModel(fit,prefit=True)
    values= SelectFromModel.get_support(sfm,indices=True)
    new_features = sfm.transform(X)
   
    return new_features,values


def get_elastic_net(X,Y,testX,testY):
    x_scaled=feature_scaling_reduction(X)
    testX_scaled=feature_scaling_reduction(testX)
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    
    
    for i in range(len(testY)):
        testY[i]=int(testY[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels_test = to_categorical(testY, num_classes=None)
    
    x_new,values=elastic_net(x_scaled,categorical_labels)
    testX_reduced=get_trim_test_data(testX_scaled,values)
    return x_new,categorical_labels,testX_reduced,categorical_labels_test
    


def select_features_lasso(X,Y):
    clf = LassoCV()
    # Set a minimum threshold of 0.25
    fit=clf.fit(X,Y)
    sfm = SelectFromModel(fit,prefit=True)
    values= SelectFromModel.get_support(sfm,indices=True)

    new_features = sfm.transform(X)
    return new_features,values


def get_lasso_features(X,Y,testX,testY):
    x_scaled=feature_scaling_reduction(X)
    testX_scaled=feature_scaling_reduction(testX)
    x_new,values=select_features_lasso(x_scaled,Y)
    testX_reduced=get_trim_test_data(testX_scaled,values)
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    
    for i in range(len(testY)):
        testY[i]=int(testY[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels_test = to_categorical(testY, num_classes=None)
    return x_new,categorical_labels,testX_reduced,categorical_labels_test


def select_features_lasso_lars(X,Y):
    clf = LassoLarsCV()
    # Set a minimum threshold of 0.25
    fit=clf.fit(X,Y)
    sfm = SelectFromModel(fit,prefit=True)
    values= SelectFromModel.get_support(sfm,indices=True)
    for i in range(0,len(values)):
        print(values[i])
    new_features = sfm.transform(X)
    return new_features,values

def get_lasso_lars_features(X,Y,testX,testY):
    x_scaled=feature_scaling_reduction(X)
    testX_scaled=feature_scaling_reduction(testX)
    x_new,values=select_features_lasso_lars(x_scaled,Y)
    testX_reduced=get_trim_test_data(testX_scaled,values)
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
   
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)
    
   
    return x_new,categorical_labels,testX_reduced,testY

    
    

def split_data_reduction(x_heart,y_heart):
    #Splitting into Trainning and test
    from sklearn.cross_validation import train_test_split
    x_train_heart,x_test_heart,y_train_heart,y_test_heart=train_test_split(x_heart,y_heart,test_size=0.10,random_state=0);    
    return x_train_heart,x_test_heart,y_train_heart,y_test_heart


from sklearn.linear_model import Ridge
def ridge_regression(x_train,y_train, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(x_train,y_train)
    y_pred = ridgereg.predict(x_train)

    
    #Return the result in pre-defined format
    rss = sum((y_pred-y_train)**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret




def main():
    X,Y=unproc()
    #split data
    for i in range(len(Y)):
        Y[i]=int(Y[i])  
    x_scaled=feature_scaling_reduction(X)
   # x_train,x_test,y_train,y_test=split_data(X,Y)
   # x_train,x_test=feature_scaling(x_train,x_test)
    #y_test=np.array(list(y_test))
  
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(Y, num_classes=None)

    fig, ax_rows = plt.subplots(2, 2, figsize=(8, 5))

    degree = 9
    alphas = [1e-3, 1e-2]
    for alpha, ax_row in zip(alphas, ax_rows):
        ax_left, ax_right = ax_row
        est = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha))
        est.fit(x_train, y_train)
        plot_approximation(x_train,y_train,est, ax_left, label='alpha=%r' % alpha)
        plot_coefficients(est, ax_right, label='Lasso(alpha=%r) coefficients' % alpha, yscale=None)

    plt.tight_layout()
   
    alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

    #Initialize the dataframe for storing coefficients.
    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,14)]
    ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
    coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

    models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
    for i in range(10):
        coef_matrix_ridge.iloc[i,] = ridge_regression(x_scaled,Y, alpha_ridge[i])
    
    coef_matrix_ridge.apply(lambda x: sum(x.values==0),axis=1)
    
    
    
    
    #Define the alpha values to test
    alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

    #Initialize the dataframe to store coefficients
    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,14)]
    ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
    coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)
    
    #Define the models to plot
    models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

    #Iterate over the 10 alpha values:
    for i in range(10):
        new_lasso = lasso_regression(x_scaled, Y, alpha_lasso[i])
        print(new_lasso.shape)
        print(new_lasso)
    
    coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)
    
    select_features_lasso(x_scaled,Y)
    
    new=linear_svc(x_scaled,Y)
    tree=tree_based(x_scaled,Y)
    
    new_univ=univ_selection(X,Y)
    new_pca=univ_selection(X,Y)
    extra_tree(X,Y)
    elastic=elastic_net(x_scaled,categorical_labels)
   

    lasso_values=select_features_lasso(x_scaled,Y)
    lasso_lars_values=select_features_lasso_lars(x_scaled,Y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
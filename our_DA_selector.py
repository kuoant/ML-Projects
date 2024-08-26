
"""
================================================================
    support for simplified model selection 
    for discriminant analysis
    with sklearn
================================================================

provided methods:

    da_model_selector( modelSpecification ): 
        input: modelSpecification = ( name of method ,  hyperparameters)
        output: sklearn model
    
    decision_surface_2d(model, X,y):
        input: fitted model, X:features (exactly 2), y:target

version 2024-03-14

"""

#%% required packages etc.

import pandas as pd
import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


#%% prepare model from sklearn
# provide tuple model = ( name of method ,  hyperparameters)

def _parse_arguments(modelSpecification):  # internal function
    # model =>  method, args=[], kwargs={}

    if type(modelSpecification) not in (list, tuple):
        return modelSpecification, [], {}
    
    method, *arguments = modelSpecification

    if len(arguments)==0:
        return method, [], {}
    
    if (len(arguments)==1) and (type(arguments[0]) in (list,tuple)):
        arguments = arguments[0]
        
    if type(arguments[-1]) == dict:
        args = arguments[:-1]
        kwargs = arguments[-1]
    else:
        args = arguments
        kwargs = {}

    return method, args, kwargs


def da_model_selector(modelSpecification):
    method, args, kwargs = _parse_arguments(modelSpecification)
               
    method = method.upper()
    
    if method=='LDA': # linear discriminant analysis
        import sklearn.discriminant_analysis as ml
        model = ml.LinearDiscriminantAnalysis(*args,**kwargs)
    
    elif method in [ 'LOGREG', 'LR', 'LOGIT']: # logistic regression
        import sklearn.linear_model as ml
        model =  ml.LogisticRegression(*args,**kwargs)
    
    elif method == 'QDA': # quadratic discriminant analysis
        import sklearn.discriminant_analysis as ml
        model = ml.QuadraticDiscriminantAnalysis(*args,**kwargs)
        
    elif method in ['NB', 'NAIVEBAYES']: # naïve Bayes
        import sklearn.naive_bayes as ml
        model = ml.GaussianNB(*args,**kwargs)
        
    elif method in ['KNN']:  # k nearest neighbors
        import sklearn.neighbors as ml
        model = ml.KNeighborsClassifier(*args,**kwargs)
            
    elif method in ['SVM', 'SVC']:  # support vector machinve, sv classification
        import sklearn.svm as ml
        model = ml.SVC(*args,**kwargs)

    elif method in ['ANN', 'NN']:  # artifical neural network, multi-layer perceptron
        import sklearn.neural_network as ml
        model = ml.MLPClassifier(*args,**kwargs)

    elif method in ['CT', 'DT']:   # decision tree, classification tree
        import sklearn.tree as ml
        model = ml.DecisionTreeClassifier(*args,**kwargs)

    elif method in ['RF', 'RandomForest']:  # random forest
        import sklearn.ensemble as ml
        model = ml.RandomForestClassifier(*args,**kwargs)

    elif method in ['ADABOOST']:   # AdaBoost
        import sklearn.ensemble as ml
        if not 'estimator' in kwargs.keys():
            base_model = ( 'DT', {'max_depth':2} )
            kwargs['estimator'] = da_model_selector(base_model)
        model = ml.AdaBoostClassifier(*args,**kwargs)
        
    elif method in ['GB', 'GRADBOOST']:  # gradient boosting
        import sklearn.ensemble as ml
        model = ml.GradientBoostingClassifier(*args,**kwargs)
        pass    
    
    elif method in ['XGB', 'XGBOOST']:  # extreme gradient boosting, XGBoost
        import xgboost as ml
        model = ml.XGBClassifier(*args,**kwargs)
        pass

    else:
        msg = f' ... add method "{method}" to your model selector ... '
        raise NotImplementedError( msg )

    return model


#%% visualization for DA problems with 2 features

def decision_surface_2d(model, X,y):

    if X.shape[1]==2:
        cmB,cmX = plt.cm.seismic, plt.cm.Dark2
        i = np.random.permutation(len(y))
        Xa, ya = np.array(X), np.array(y)
        DecisionBoundaryDisplay.from_estimator(model,X,cmap=cmB,alpha=.3)
        plt.scatter(*Xa[i].T,c=pd.factorize(ya)[0][i], cmap=cmX, 
                    edgecolor='w',alpha=.5)
        plt.title(model)
        plt.show()
    else:
        print('>>>>>> sorry, "decision_surface_2d()" requires exactly 2 features ... \n')

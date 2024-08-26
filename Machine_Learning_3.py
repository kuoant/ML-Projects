#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:28:21 2024

@author: fabiankuonen
"""

"""
DISCRIMINANT ANALYSIS -- GENERAL
"""

#%% modules


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn as skl
import sklearn.linear_model
import sklearn.discriminant_analysis
import sklearn.neighbors
import sklearn.neural_network
from our_DA_selector import da_model_selector, decision_surface_2d
from sklearn.tree import plot_tree


#%% model selection

def da_model_selector(method):
    
    if method == "lda":
        model = skl.discriminant_analysis.LinearDiscriminantAnalysis()
    
    elif method in ["lr", "logit", "log", "logreg"]:
        model = skl.linear_model.LogisticRegression()
    
    elif method == "qda":
        model = skl.discriminant_analysis.QuadraticDiscriminantAnalysis()
        
    elif method == "svm":
        model = skl.svm.SVC(kernel="rbf")
        
    elif method == "knn":
        model = skl.neighbors.KNeighborsClassifier(n_neighbors=10)
        
    elif method == "ann":
        model = skl.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver = "adam", max_iter=1900)
        
    elif method == "dt":
        model = skl.tree.DecisionTreeClassifier(max_depth=5, min_samples_split=10)
        
    elif method == "rf":
        model = skl.ensemble.RandomForestClassifier(n_estimators=100)
        
    elif method == "nb":
        model = skl.naive_bayes.GaussianNB()
    
    return model


#%% data selection

# Set 01 violates the assumption that we need same variance
# Set 02 we need two axes to separate data, because we have three variables
data = pd.read_csv("data/class_08.csv")
y = data["G"]
X = data.drop("G", axis=1)

dfC = pd.concat( (X, y), axis=1 )

sns.pairplot(dfC, hue="G")


#%% experiments

# Linear: LDA, Logit, QDA
# Non-linear: NB, SVM, ANN
# Tree-based: DT, RF
# Instance-based: KNN

method ="dt"
model = da_model_selector(method)

train_list = []
test_list = []

for it in range(10):
    
    # picks 70% randomly in the train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model.fit(X_train, y_train) 
    
    accIS = model.score(X_train, y_train)
    accOS = model.score(X_test, y_test)
    
    train_list.append(accIS)
    test_list.append(accOS)
    
    print(it)
    
print(np.mean(train_list), np.mean(test_list))

decision_surface_2d(model, X_train, y_train)
plot_tree(model, feature_names=X.columns)

accIS = model.score(X_train, y_train)
accOS = model.score(X_test, y_test)

print(accIS, "vs", accOS)



# we don't want have a very high in-sample accuracy and bad out-sample accuracy
# need to trade-off by experience





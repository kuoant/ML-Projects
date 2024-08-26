#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:29:22 2024

@author: fabiankuonen
"""

import numpy as np
import pandas as pd
import seaborn as sns
from our_DA_selector import da_model_selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import PowerTransformer

#%% Load data

# Load incomplete data
na_symbols = ["NA", "MISSING", "?", "NULL"]
data = pd.read_csv("data/crx_data.csv", sep=";", na_values=na_symbols)

data = data.dropna() # get rid of incomplete data rows

y = data["A16"]
X = data.drop("A16", axis=1)
X = pd.get_dummies(X)

y = y.replace("+", 0)
y = y.replace("-", 1)
y = y.astype("category")

sns.pairplot(data=data, hue="A16")

#%%


# Run through decision trees and random forests, too many dummies bad for ANNs
train_list = []
test_list = []


for i in range(10):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    
    # Scaling
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train[:] = scaler.transform(X_train)
    X_test[:]  = scaler.transform(X_test)
    
    
    # Transforming, does not work
    # transformer = PowerTransformer("yeo-johnson")
    # transformer.fit(X_train)
    # X_train[:] = transformer.transform(X_train)
    # X_test[:] = transformer.transform(X_test)
    
    
    select = "dt"
    model = da_model_selector(select)
    model.fit(X_train, y_train)
    
    accIS = model.score(X_train, y_train)
    accOS = model.score(X_test, y_test)
    
    train_list.append(accIS)
    test_list.append(accOS)
    
    print(i)
    
    

y_pred = model.predict(X_train)
print("In-sample CM:  \n", confusion_matrix(y_train, y_pred))
    
y_pred = model.predict(X_test)
print("Out-sample CM: \n", confusion_matrix(y_test, y_pred))
    
print(np.mean(train_list), np.mean(test_list))





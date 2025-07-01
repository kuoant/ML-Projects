#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:15:37 2024

"""

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from our_DA_selector import da_model_selector
import sklearn.preprocessing as prep


#%% Load & Describe Data

data = pd.read_csv("data/_wholesale_cat.csv")
X = data.drop(["Channel", "Region"], axis=1)
y = data["Channel"]

sns.pairplot(data, hue="Channel")

# Not normally distributed, want to apply transformation
print(X.skew())
print(X.kurt())


#%% Preprocess Data

# split into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X,y)


# transform data to reduce skewness, transformation changes the shape!
transformer = prep.PowerTransformer(method="yeo-johnson")
transformer.fit(X_train)
X_train[:] = transformer.transform(X_train)
X_test[:] = transformer.transform(X_test)

# Get rid of outliers, change the distribution 
# Problem cannot interpret this, but it helps to fit the straight line
# preserves some information about the order, but distances are distorted
# downside of transforming: it has no meaning anymore
# if we don't have the transformer, we cannot use the method anymore
# So we just use it if it's necessary

sns.pairplot(X_train.join(y_train), hue="Channel")


#%% Scaling

# Scaling (Standardizing) 
# scaler = prep.Standardscaler()
# scaler.fit(X_train)
# X_train[:] = scaler.transform(X_train)
# X_test[:] = scaler.transform(X_test)

# Then we have the same distribution but differently scaled
# This is what we want to do so far, in order to make Betas more understandable
# and help the ANN

# Scaling (Normalization)
# scaler = prep.MinMaxScaler()
# Also no influence on distribution, just differently scaled
# need for e.g. Clustering


#%% Fit Classifier and Predict

df = pd.read_csv("data/class_03.csv")

sns.pairplot(df, hue="G")

X = df.drop("G", axis=1)
y = df["G"]


#%%

train_list = []
test_list = []

for i in range(10):

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    # Scaling
    scaler = prep.StandardScaler()
    scaler.fit(X_train)
    X_train[:] = scaler.transform(X_train)
    X_test[:] = scaler.transform(X_test)
    

    #  LDA, Logit, QDA, NB, SVM, KNN, ANN, DT, Ensemble (RF)
    choice = ("dt", {"max_depth":2})
    model = da_model_selector(choice)
    model.fit(X_train, y_train)
    
    accIS = model.score(X_train, y_train)
    accOS = model.score(X_test, y_test)
    
    train_list.append(accIS)
    test_list.append(accOS)
    
    print(i)

print(np.mean(train_list), np.mean(test_list))

from sklearn.tree import plot_tree

plot_tree(model, feature_names=X_train.columns)



















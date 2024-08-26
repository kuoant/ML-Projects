#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:56:36 2024

@author: fabiankuonen

@title: Linear Discriminant Analysis, Banknotes
"""

#%% load packages

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as skl

# %matplotlib qt #creates bigger pictures
# %matplotlib inline # back to default

#%% get data

data = pd.read_csv("data/banknote_authentication.csv", sep=";")

# in pandas we can change it to categorical data, so that it is a label and not just a number
data["class"] = data["class"].astype("category") 

# Now watch out for features we can rotate to separate
sns.pairplot(data, hue="class", plot_kws={"alpha":.1})

#%% LDA

X = data.loc[:, ["skewness", "variance"]]  # features, which we could separate
y = data["class"]                          # want to explain fake/real banknotes

sns.scatterplot(data, x="skewness", y="variance", hue="class")

# Model in sklearn is an object, which contains data & statistics
# In our case we have an object that performs LDA, operate on the model

import sklearn.discriminant_analysis as da

# Create the LDA object
model = da.LinearDiscriminantAnalysis()

# train the model, in the background it is fitted
model.fit(X, y)

# get weights a, everything that ends with _ is something sklearn estimated
model.coef_

# get group means, 2x2 because we are in the coordinate system with two directions
# Means separated by labels
print(model.means_)

# get prior probabilities
model.priors_

# get confusion matrix
y_pred = model.predict(X)
cm = skl.metrics.confusion_matrix(y,y_pred)

# accuracy
print(np.diag(cm).sum() / cm.sum())
acc = model.score(X, y)
print(f"accuracy = {acc:0.3} ")
# We increased probability from roughly 55% to nearly 90%
# Of course we can increase that by adding more features

# We expect entropy to contribute very small amount of explaining the data
model.coef_
X.std()
# But we must check the standard deviation first, because it could be that it
# explains much even though coefficient is small





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:49:58 2024

@author: fabiankuonen
"""

import matplotlib.pyplot as plt
import pandas as pd
import AML_support as aml
from sklearn.decomposition import PCA


#%% Data

data = pd.read_csv("data/Switzerland_population.csv", sep=";", index_col=0)
data = data.loc[:, 'Lang_German':'noConfession']
data = ( data - data.mean() ) / data.std()

aml.scatter(data)
aml.correlogram(data)


#%% PCA

model = PCA()
score = model.fit_transform(data)

# components = eigenvectors can be positive or negative, does not matter
# eigenvectors are not unique
print(model.components_)

# score shows us which representation is best
print(score.var(0) / score.var(0).sum())

# get the eigenvalues = explained variance in the context of PCA
print(model.explained_variance_)

# variance ratio contained by components, should be as large as possible
print(model.explained_variance_ratio_)


# components tell us: if one variable enter positive and one enters negative
# they are negatively correlated

# Biplot = best 2D representation consisting of the first two components only
aml.biplot(data)

# If age arrows point in the same direction, we have correlation.
# Long arrow means important features and if small then not relevant.

# E.g. Basel is in the opposite direction of young people.
# We can see the in the low dimension representation condensed information.
# Observations which are close are very similar. 



# use %matplotlib qt, otherwise does not work
for i, canton in enumerate(data.index):
    plt.text(score[i, 0], score[i,1], canton)

# PCA = Information is measured in terms of variation.






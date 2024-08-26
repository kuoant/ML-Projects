#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:45:35 2024

@author: fabiankuonen
"""

import pandas as pd
import sklearn.metrics as metrics
import seaborn as sns
import sklearn.cluster as cluster
import AML_support as aml
import sklearn.preprocessing as prep


#%% Euclidean Distance
data = pd.read_csv("data/fitness.csv", sep=";")

# Measures distance between two observations across all their features
metrics.euclidean_distances(data)


#%% Load Data

# how much time did readers spend time on sports, front, gossip pages
# data = pd.read_csv("data/onlinenews.csv", sep=";")

# This is a case for classical clustering, not hierarchical clustering
data = pd.read_csv("data/clust_06.csv")


#%% Partitioned Clustering K-means

# sns.pairplot(data)
# aml.scatter(data)

# Sports can be splitted away (interpreted with means, high values indicate it)
# 1. group which spend high on sports 0.66 vs. low values 
# 2. front local gossip and 
# 3. group who reads economics, politics

model = cluster.KMeans(n_clusters = 3, n_init=100)
model.fit(data)

# all centers, 3D coordinates for first group, 3d coordinates for second group
print(model.cluster_centers_)


# Visualize predicted labels
labels_ = model.labels_
sns.pairplot(data, plot_kws = {"hue": labels_})
aml.scatter(data, labels_)

# average silhouette score should be as high as possible,
# means that most observations are in the right group
# few negatives (wrong groups) are fine, because otherwise we would
# change groups and had to calculate new means which has follow up wrongs
# vary number of means and check silhouette, highest value for 3 clusters 
aml.silhouette_plot(data, labels_)


#%% Hierarchical Clustering AGNES

data = pd.read_csv("data/Switzerland_population.csv", sep=";", index_col=0)
X = data.filter(regex="Pop")

import scipy.cluster.hierarchy as sch

# linkage: single or complete
clusters = sch.linkage(X, method = "complete", metric = "euclidean")


# Depending on data, dendrogram intrepretation easy (language) or hard (age)
dg = sch.dendrogram(clusters, labels = X.index)



#%% Density-Based Clustering DBSCAN

data = pd.read_csv("data/clust_06.csv")

# if data different scales, standardize first to make sense for eps
X = data.copy()
scaler = prep.StandardScaler()
X[:] = scaler.fit_transform(X)

aml.scatter(X)
sns.pairplot(X)

core_sample, labels_ = cluster.dbscan(X, eps=0.6) # eps = radius

aml.scatter(X, labels_)

sns.pairplot(X, plot_kws = {"hue":labels_})

aml.silhouette_plot(data, labels_)

#%% Experiments

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import sklearn


data = pd.read_csv("data/clust_01.csv")


X = data.copy()
scaler = prep.StandardScaler()
scaler.fit(X)
X[:] = scaler.transform(X)


# DBSCAN
model, labels = sklearn.cluster.dbscan(X, eps = 0.6)
sns.pairplot(X, plot_kws={"hue":labels})
aml.silhouette_plot(data, labels)


# K-Means
model = sklearn.cluster.KMeans(n_clusters=3, n_init=50)
model.fit(X)
labels = model.labels_

sns.pairplot(X, plot_kws={"hue":labels})
aml.silhouette_plot(data, labels)


# AGNES
model = sch.linkage(X, method="single", metric="euclidean")
sch.dendrogram(model, labels = X.index)


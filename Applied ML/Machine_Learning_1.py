#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:30:22 2024


"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%% Dataset Switzerland

df_orig_CH = pd.read_csv("data/Switzerland_Population.csv", sep=";", index_col=0)

# Copy the original
df = df_orig_CH.copy()

# Access a feature, get the Series
df["Population"]
df[["Population", "Age0-19"]]

# Sort by a specific feature
df.sort_values(by="Population")

# Refer to index using .loc[]
df.loc["BS"]
df.loc[["BS", "ZH"], ["Population", "FractNonSwiss"]]

# Using the iloc for row and col numpy index
df.iloc[7,5] # row 7, col 5

# Some information about the data set
df.head(3) # first observations
df.tail(3)
df.describe()
df.columns # available features, names

# Create a new feature my_Area and "Canton"
df["my_Area"] = df["Population"] / df["PopPersqkm"]
df.insert(0,"Canton", df.index) #column 0, called "Canton", values=index
df.columns

#%% Matplotlib Visualization

dfr = df.loc[:, "Age0-19":"Age64plus"] # exception: includes the last element
dfr.hist()
plt.show()


#%% Seaborn Visualization

sns.pairplot(dfr)


#%% Data Wrangling (modify the shape of the data)

dfr = df_orig_CH.loc[:"BS", "Lang_German":"Lang_Ital"]
dfr["Canton"] = dfr.index # create new feature with Canton name

dfm = dfr.melt(id_vars="Canton", 
               value_vars=["Lang_German", "Lang_French", "Lang_Ital"], 
               var_name="Language", value_name="share")

dfm["Language"] = dfm["Language"].str.replace("Lang_", "")

print(dfm)

# Using Pivoting
dfp = dfm.pivot(index="Canton",  # row: based on column "Canton
                columns="Language", # create one col (feature per language)
                values  = "share")

print(dfp)

dfm.sort_values(by="Canton", ascending=False)

#%% Dataset Banknote Authentication

df = pd.read_csv("data/Banknote_authentication.csv", sep=";")

# Statistics
df.head()
df.describe()
df.mean()
df.std()
df.max()
df.median()
df.idxmax()
df.idxmin() # observation 1233 has the highest variance

#%% Pairplot

# Cannot use alpha to change color
# because pairplot is a plot on a plot, use keywords kws instead
sns.pairplot(df, hue="class", plot_kws={"alpha":0.1}) # Seaborn is slow
plt.show()

#%% Kernel Density

# Kernel Density, small Gaussian bells, smooth histogram
sns.distplot(df["variance"])

#%% Violin Plot

# Fancy boxplot called Violin Plot, kernel density mirrored 
sns.violinplot(df, x="class", y="variance", inner="quart", hue="class", split=True)

#%%

# Plot the transpose, ploting the rows, each line is one observation
plt.plot(df.T, "k", alpha=0.02)
plt.show()

# Negative correlation is when observations jump from high to low values 
# (e.g. skewness and kurtosis)
one = df["class"] == 0

plt.plot(df[~one].T, "k", alpha=0.02)
plt.plot(df[one].T, "r", alpha=0.02)
plt.legend(["one", "not one"])




















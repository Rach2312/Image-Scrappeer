#!/usr/bin/env python
# coding: utf-8

# Load the dataset into a Pandas dataframe.
# Split the dataset into features and target variables.
# Perform data preprocessing (e.g., scaling, normalisation, missing value imputation) as necessary.
# Implement PCA on the preprocessed dataset using the scikit-learn library.
# Determine the optimal number of principal components to retain based on the explained variance ratio.
# Visualise the results of PCA using a scatter plot.
# Perform clustering on the PCA-transformed data using K-Means clustering algorithm.
# Interpret the results of PCA and clustering analysis.
# 
# Load the dataset into a Pandas dataframe.
# Split the dataset into features and target variables.
# Perform data preprocessing (e.g., scaling, normalisation, missing value imputation) as necessary.
# Implement PCA on the preprocessed dataset using the scikit-learn library.
# Determine the optimal number of principal components to retain based on the explained variance ratio.
# Visualise the results of PCA using a scatter plot.
# Perform clustering on the PCA-transformed data using K-Means clustering algorithm.
# Interpret the results of PCA and clustering analysis.
# ChatGPT

# In[11]:


import pandas as pd

wine_df = pd.read_csv(r'C:\Users\dvkha\Downloads\wine.data', header=None)

X = wine_df.iloc[:, 1:]
y = wine_df.iloc[:, 0]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
import matplotlib.pyplot as plt

plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
plt.show()

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_pca)


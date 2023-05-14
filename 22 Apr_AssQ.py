#!/usr/bin/env python
# coding: utf-8

# Write a Python code to implement the KNN classifier algorithm on load_iris dataset in
# sklearn.datasets.

# In[3]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Write a Python code to implement the KNN regressor algorithm on load_boston dataset in
# sklearn.datasets.

# In[ ]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

k = 5
knn = KNeighborsRegressor(n_neighbors=k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"KNN regression (k={k})")
print(f"Mean squared error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# Write a Python code snippet to find the optimal value of K for the KNN classifier algorithm using
# cross-validation on load_iris dataset in sklearn.datasets.

# In[5]:


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

k_range = list(range(1, 31))

cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_range[cv_scores.index(max(cv_scores))]
print("The optimal number of neighbors is %d" % optimal_k)


# Implement the KNN regressor algorithm with feature scaling on load_boston dataset in
# sklearn.datasets.

# In[ ]:



from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Write a Python code snippet to implement the KNN classifier algorithm with weighted voting on
# load_iris dataset in sklearn.datasets.

# In[ ]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# Implement a function to standardise the features before applying KNN classifier.

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def knn_classifier_with_standardization(X_train, y_train, X_test, k):
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    

    y_pred = knn.predict(X_test_scaled)
    
    return y_pred


# Write a Python function to calculate the euclidean distance between two points.

# In[9]:


import math

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


# Write a Python function to calculate the manhattan distance between two points.

# In[10]:


import numpy as np

def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan distance between two points.

    Parameters:
    point1 (array-like): A point represented as an array-like object.
    point2 (array-like): A point represented as an array-like object.

    Returns:
    The Manhattan distance between the two points.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sum(np.abs(point1 - point2))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# Build a random forest classifier to predict the risk of heart disease based on a dataset of patient
# information. The dataset contains 303 instances with 14 features, including age, sex, chest pain type,
# resting blood pressure, serum cholesterol, and maximum heart rate achieved.
# Dataset link: https://drive.google.com/file/d/1bGoIE4Z2kG5nyh-fGZAJ7LH0ki3UfmSJ/view?
# usp=share_link
# Q1. Preprocess the dataset by handling missing values, encoding categorical variables, and scaling the
# numerical features if necessary.

# In[22]:


import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/dvkha/Downloads/dataset.csv")

data.isnull().sum()

data["sex"] = pd.get_dummies(data["sex"], drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data[["age", "trestbps", "chol", "thalach"]] = scaler.fit_transform(data[["age", "trestbps", "chol", "thalach"]])


# Split the dataset into a training set (70%) and a test set (30%).

# In[23]:


from sklearn.model_selection import train_test_split

X = data.drop("target", axis=1)
y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Train a random forest classifier on the training set using 100 trees and a maximum depth of 10 for each
# tree. Use the default values for other hyperparameters.

# In[24]:


from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

rf_classifier.fit(X_train, y_train)


# Evaluate the performance of the model on the test set using accuracy, precision, recall, and F1 score.

# In[25]:


from sklearn import metrics

y_pred = rf_classifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)

precision = metrics.precision_score(y_test, y_pred)

recall = metrics.recall_score(y_test, y_pred)

f1_score = metrics.f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# Use the feature importance scores to identify the top 5 most important features in predicting heart
# disease risk. Visualise the feature importances using a bar chart.

# In[27]:


import matplotlib.pyplot as plt

importances = rf_classifier.feature_importances_

indices = np.argsort(importances)[::-1][:5]

print("Top 5 most important features:")
for i in indices:
    print(X.columns[i], ":", importances[i])

plt.figure(figsize=(10, 5))
plt.title("Feature Importances")
plt.bar(range(5), importances[indices])
plt.xticks(range(5), X.columns[indices], rotation=90)
plt.show()


# Tune the hyperparameters of the random forest classifier using grid search or random search. Try
# different values of the number of trees, maximum depth, minimum samples split, and minimum samples
# leaf. Use 5-fold cross-validation to evaluate the performance of each set of hyperparameters.

# In[28]:


from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


rf_classifier = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


# Report the best set of hyperparameters found by the search and the corresponding performance
# metrics. Compare the performance of the tuned model with the default model.

# In[32]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


data = pd.read_csv('C:/Users/dvkha/Downloads/dataset.csv')

X = data.drop('target', axis=1)
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(rf, param_grid, cv=5)


grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)

tuned_score = grid_search.score(X_test, y_test)
print("Tuned model accuracy:", tuned_score)


default_model = RandomForestClassifier(random_state=42)
default_model.fit(X_train, y_train)
default_score = default_model.score(X_test, y_test)
print("Default model accuracy:", default_score)

if tuned_score > default_score:
    print("The tuned model performs better than the default model.")
else:
    print("The default model performs better than the tuned model.")


# Interpret the model by analysing the decision boundaries of the random forest classifier. Plot the
# decision boundaries on a scatter plot of two of the most important features. Discuss the insights and
# limitations of the model for predicting heart disease risk.

# In[34]:



rf = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=10, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
feature_names = X.columns
most_important_idx = importances.argsort()[-2:]
most_important_names = feature_names[most_important_idx]
X_most_important = X[most_important_names]

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


df = pd.read_csv("C:/Users/dvkha/Downloads/dataset.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10]
}


rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X, y)

print("Best hyperparameters:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

default_rf = RandomForestClassifier(random_state=42)
default_rf.fit(X, y)
default_acc = accuracy_score(y, default_rf.predict(X))

tuned_rf = grid_search.best_estimator_
tuned_rf.fit(X, y)
tuned_acc = accuracy_score(y, tuned_rf.predict(X))

print("Default accuracy score:", default_acc)
print("Tuned accuracy score:", tuned_acc)


# In[35]:



plt.scatter(X_most_important.iloc[:, 0], X_most_important.iloc[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#00FF00']))
plt.xlabel(most_important_names[0])
plt.ylabel(most_important_names[1])
plt.title('Scatter Plot of Two Most Important Features')
plt.show()


# In[ ]:





# In[ ]:





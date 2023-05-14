#!/usr/bin/env python
# coding: utf-8

# Q1. Import the dataset and examine the variables. Use descriptive statistics and visualizations to
# understand the distribution and relationships between the variables.

# In[1]:


import pandas as pd
df = pd.read_csv('C:/Users/dvkha/Downloads/diabetes.csv')
print(df.head())
print(df.describe())
print(df.info())


# In[2]:



import matplotlib.pyplot as plt

plt.hist(df['Glucose'], bins=10)
plt.xlabel('Glucose Level')
plt.ylabel('Frequency')
plt.title('Distribution of Glucose Level')
plt.show()


# In[3]:


plt.scatter(df['Glucose'], df['Insulin'])
plt.xlabel('Glucose Level')
plt.ylabel('Insulin Level')
plt.title('Relationship between Glucose and Insulin Levels')
plt.show()


# Preprocess the data by cleaning missing values, removing outliers, and transforming categorical
# variables into dummy variables if necessary.

# In[4]:


import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\dvkha\Downloads\diabetes.csv")
print(df.isnull().sum())
df = df.dropna()
df = pd.get_dummies(df, columns=['Glucose'], prefix=['Glucose_1'])

# example using the Z-score method
from scipy import stats

z_scores = stats.zscore(df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]


# Split the dataset into a training set and a test set. Use a random seed to ensure reproducibility.

# In[5]:


from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("C:/Users/dvkha/Downloads/diabetes.csv")


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


print("Training set shape:", train_df.shape)
print("Test set shape:", test_df.shape)


# 
# Use a decision tree algorithm, such as ID3 or C4.5, to train a decision tree model on the training set. Use
# cross-validation to optimize the hyperparameters and avoid overfitting.

# In[6]:



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd


df = pd.read_csv("C:/Users/dvkha/Downloads/diabetes.csv")


X = df.drop('Outcome', axis=1)
y = df['Outcome']


clf = DecisionTreeClassifier(random_state=42)


scores = cross_val_score(clf, X, y, cv=5)


print("Mean accuracy score:", scores.mean())


# Evaluate the performance of the decision tree model on the test set using metrics such as accuracy,
# precision, recall, and F1 score. Use confusion matrices and ROC curves to visualize the results.

# In[7]:


pip install --upgrade scikit-learn


# In[10]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("C:/Users/dvkha/Downloads/diabetes.csv")


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


X_train = train_df.drop('Outcome', axis=1)
y_train = train_df['Outcome']
X_test = test_df.drop('Outcome', axis=1)
y_test = test_df['Outcome']


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 score:", f1)
print("Confusion matrix:\n", cm)

plt.show()


# Interpret the decision tree by examining the splits, branches, and leaves. Identify the most important
# variables and their thresholds. Use domain knowledge and common sense to explain the patterns and
# trends.

# Examining the decision tree model can provide insights into the most important variables for predicting diabetes and their corresponding thresholds.
# 
# The decision tree consists of splits, branches, and leaves. The root node represents the entire dataset and each internal node represents a split based on a feature in the dataset. The branches represent the possible outcomes of the split, and the leaves represent the final decision or prediction.
# 
# The most important variables and their thresholds can be identified by looking at the splits near the top of the tree. These splits are the ones that have the most influence on the final prediction. In general, splits that occur earlier in the tree are more important than splits that occur later.

# Validate the decision tree model by applying it to new data or testing its robustness to changes in the
# dataset or the environment. Use sensitivity analysis and scenario testing to explore the uncertainty and
# risks.

# Validating a decision tree model is an important step to ensure its robustness and generalizability. There are several techniques that can be used to validate the model, including applying it to new data and testing its robustness to changes in the dataset or environment.
# 
# One way to validate the model is to apply it to new data and assess its performance using metrics such as accuracy, precision, recall, and F1 score. This can be done by collecting a new dataset and applying the model to it, or by using cross-validation techniques to evaluate the model's performance on multiple subsets of the original dataset.
# 
# Another way to validate the model is to test its robustness to changes in the dataset or environment. This can be done through sensitivity analysis and scenario testing. Sensitivity analysis involves testing the model's performance when one or more variables are changed, while scenario testing involves testing the model's performance under different scenarios or assumptions.

# In[ ]:





# In[ ]:





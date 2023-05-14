#!/usr/bin/env python
# coding: utf-8

# What is the relationship between polynomial functions and kernel functions in machine learning
# algorithms?

# Polynomial functions and kernel functions are related in the sense that kernel functions can be used to implicitly compute the dot product between the feature vectors of input data in a higher-dimensional space, which is equivalent to applying a polynomial function to the original input data.
# 
# In machine learning algorithms, kernel functions are used as a technique to transform the input data into a higher-dimensional feature space. This allows the data to be represented in a more complex way, making it possible to find more complex patterns in the data that would not be visible in the original feature space.

# How can we implement an SVM with a polynomial kernel in Python using Scikit-learn?

# In[13]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


iris = load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm = SVC(kernel='poly', degree=3)

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# How does increasing the value of epsilon affect the number of support vectors in SVR?

# In support vector regression (SVR), epsilon is a hyperparameter that controls the width of the margin around the regression line where no penalty is given to errors. Increasing the value of epsilon allows for more errors to be tolerated, which results in a wider margin and potentially more support vectors.
# 
# As the value of epsilon increases, the number of support vectors in SVR tends to increase as well. This is because increasing epsilon results in a wider margin around the regression line, which allows more data points to be inside the margin and therefore contribute to the definition of the support vectors.

# How does the choice of kernel function, C parameter, epsilon parameter, and gamma parameter
# affect the performance of Support Vector Regression (SVR)? Can you explain how each parameter works
# and provide examples of when you might want to increase or decrease its value?

# how each parameter works and how it can affect the performance of the SVR model:
# 
# Kernel function: The kernel function is used to transform the input features into a higher-dimensional space, where it may be easier to find a linear separation between the data points. Common kernel functions include the linear kernel, the polynomial kernel, and the radial basis function (RBF) kernel. The choice of kernel function can affect the performance of the SVR model by determining the shape of the decision boundary. For example, the linear kernel may work well if the data is linearly separable, while the RBF kernel may work well if the data is nonlinear.
# 
# C parameter: The C parameter controls the trade-off between achieving a low training error and a low testing error. A smaller C value will result in a wider margin and allow for more errors, while a larger C value will result in a narrower margin and fewer errors. Increasing the value of C may result in better performance on the training data, but may also lead to overfitting.
# 
# Epsilon parameter: The epsilon parameter determines the width of the margin around the predicted values where no penalty is given to errors. A smaller epsilon value will result in a narrower margin and fewer errors, while a larger epsilon value will result in a wider margin and more errors. Increasing the value of epsilon may result in a more generalized model that performs better on new, unseen data.
# 
# Gamma parameter: The gamma parameter controls the smoothness of the decision boundary. A smaller gamma value will result in a smoother decision boundary, while a larger gamma value will result in a more complex decision boundary. Increasing the value of gamma may result in better performance on the training data, but may also lead to overfitting.

# Import the necessary libraries and load the dataseg
# L Split the dataset into training and testing setZ
# L Preprocess the data using any technique of your choice (e.g. scaling, normaliMationK
# L Create an instance of the SVC classifier and train it on the training datW
# L hse the trained classifier to predict the labels of the testing datW
# L Evaluate the performance of the classifier using any metric of your choice (e.g. accuracy,
# precision, recall, F1-scoreK
# L Tune the hyperparameters of the SVC classifier using GridSearchCV or RandomiMedSearchCV to
# improve its performanc_
# L Train the tuned classifier on the entire dataseg
# L Save the trained classifier to a file for future use.

# In[14]:



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svc = SVC()
svc.fit(X_train, y_train)


y_pred = svc.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 1, 10]}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best accuracy:", grid.best_score_)


svc_tuned = SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'], gamma=grid.best_params_['gamma'])
svc_tuned.fit(X, y)


joblib.dump(svc_tuned, 'svc_tuned.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





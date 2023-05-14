#!/usr/bin/env python
# coding: utf-8

# What is the mathematical formula for a linear SVM?

# w^T x + b = 0
# 
# where w is the normal vector to the hyperplane, b is the bias term, and x is the input feature vector. 

# What is the objective function of a linear SVM?

# The objective function of a linear SVM is to find the values of the weight vector w and the bias term b that maximize the margin between the hyperplane and the closest data points from each class. The margin is defined as the distance between the hyperplane and the closest data points, also known as the support vectors.
# 
# In the case of a linearly separable binary classification problem, the objective function can be formulated as a constrained optimization problem:
# 
# minimize 1/2 ||w||^2 subject to yi(w^T xi + b) ≥ 1 for all i
# 
# where ||w|| is the L2 norm of the weight vector w, yi is the class label of the ith training example, xi is the corresponding input feature vector, and b is the bias term.

# What is the kernel trick in SVM?

# The kernel trick in SVM is a technique used to extend linear SVMs to handle nonlinearly separable data by mapping the input features to a higher-dimensional space using a kernel function. This allows the SVM to find a nonlinear decision boundary that separates the data points in the transformed feature space.

# What is the role of support vectors in SVM Explain with example

# The role of support vectors is twofold. First, they determine the position and orientation of the hyperplane by defining the margin, which is the distance between the hyperplane and the closest data points. The SVM algorithm seeks to maximize this margin, and the support vectors are the data points that lie on the margin or contribute to its computation.
# 
# Second, the support vectors are used to classify new data points. The decision function of the SVM is based on the dot product between the weight vector w and the input feature vector x, and the bias term b:
# 
# f(x) = sign(w^T x + b)
# 
# 
# For example, consider a binary classification problem where the goal is to separate two classes of data points using a linear SVM. The figure below shows a scatter plot of the data points, where the blue points belong to one class and the red points belong to the other class.
# 
# 

# Illustrate with examples and graphs of Hyperplane, Marginal plane, Soft margin and Hard margin in
# SVM?

# Hyperplane:
# A hyperplane is a linear decision boundary that separates the data points in an SVM. In a two-dimensional feature space, a hyperplane is a line, while in a three-dimensional feature space, it is a plane. Here is an example of a hyperplane in a two-dimensional feature space separating two classes of data points:
# 
# Hyperplane Example Image
# 
# Marginal plane:
# The marginal plane is the boundary region on either side of the hyperplane that separates the support vectors from the rest of the data points. In a two-dimensional feature space, the marginal plane is a pair of parallel lines that run along the margin, while in a three-dimensional feature space, it is a pair of parallel planes. Here is an example of a marginal plane in a two-dimensional feature space:
# 
# 
# Hard margin:
# In a hard-margin SVM, the goal is to find a hyperplane that perfectly separates the data points of different classes with no misclassifications. This can only be achieved if the data is linearly separable. Here is an example of a hard-margin SVM in a two-dimensional feature space with linearly separable data:
# 
# 
# Soft margin:
# In a soft-margin SVM, the goal is to find a hyperplane that separates the data points with some misclassifications, allowing for a more flexible decision boundary. The degree of flexibility is controlled by a parameter C, which penalizes the misclassification of data points. A higher value of C results in a less flexible decision boundary, while a lower value of C results in a more flexible decision boundary. Here is an example of a soft-margin SVM in a two-dimensional feature space with non-linearly separable data:
# 
# 
# In the above example, the blue data points and red data points cannot be perfectly separated by a hyperplane. A soft-margin SVM is used to find the best decision boundary with some misclassifications. The yellow line represents the decision boundary of the SVM, and the dotted lines represent the marginal planes. The gray circles represent the support vectors. The value of C controls the degree of misclassification allowed, and a higher value of C results in fewer misclassifications but a less flexible decision boundary.
# 
# 
# 
# 
# 
# 

# SVM Implementation through Iris dataset.
# 
# ~ Load the iris dataset from the scikit-learn library and split it into a training set and a testing setl
# ~ Train a linear SVM classifier on the training set and predict the labels for the testing setl
# ~ Compute the accuracy of the model on the testing setl
# ~ Plot the decision boundaries of the trained model using two of the featuresl
# ~ Try different values of the regularisation parameter C and see how it affects the performance of
# the model.

# In[11]:



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()


X_train, X_test, y_train, y_test = train_test_split(iris.data[:, [0, 2]], iris.target, test_size=0.2, random_state=42)


svm = SVC(kernel='linear')
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('Linear SVM on Iris Dataset')
plt.show()


for c in [0.1, 1, 10, 100]:
    svm = SVC(kernel='linear', C=c)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("C = {}, Accuracy: {}".format(c, accuracy))


# Implement a linear SVM classifier from scratch using Python and compare its
# performance with the scikit-learn implementation.

# In[12]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LinearSVM:
    def __init__(self, learning_rate=0.001, C=1, n_iters=1000):
        self.lr = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.C * self.w)
                else:
                    self.w -= self.lr * (2 * self.C * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
                    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
    

iris = load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm = LinearSVM()
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# 

# 

# 

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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





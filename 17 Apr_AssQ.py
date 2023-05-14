#!/usr/bin/env python
# coding: utf-8

# What is Gradient Boosting Regression?

# Gradient Boosting Regression (GBR) is a type of boosting algorithm that is used for regression problems. It works by iteratively improving a regression model by adding weak models, such as decision trees, that correct the errors of the previous model.
# 
# The GBR algorithm follows these steps:
# 
# Initialize the model: The GBR algorithm starts by initializing a simple regression model, such as a constant value or a linear regression model, which will serve as the first weak model.
# 
# Compute the residuals: The difference between the predicted values of the current model and the actual values of the training data is computed, which gives the residuals.
# 
# Train a weak model on the residuals: A weak model, such as a decision tree, is trained on the residuals of the current model, rather than on the original target values. The weak model tries to predict the residuals of the current model, rather than the target values.
# 
# Update the model: The predictions of the weak model are added to the predictions of the current model, which produces an updated model. The updated model tries to correct the errors of the previous model by reducing the residuals.
# 
# Repeat steps 2-4: Steps 2-4 are repeated for a fixed number of iterations or until a threshold performance level is reached. In each iteration, a new weak model is trained on the residuals of the current model, and its predictions are added to the predictions of the previous models.
# 
# Compute the final predictions: The final predictions are computed by adding the predictions of all the weak models. The weights of the weak models are determined by their performance on the training data, which ensures that the stronger models have more influence on the final predictions.
# 
# 

# Implement a simple gradient boosting algorithm from scratch using Python and NumPy. Use a
# simple regression problem as an example and train the model on a small dataset. Evaluate the model's
# performance using metrics such as mean squared error and R-squared.

# In[1]:


import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.intercept = None

    def fit(self, X, y):
        
        self.intercept = np.mean(y)
        
        residuals = y - self.intercept
    
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            residuals = y - self.predict(X)

    def predict(self, X):
        predictions = np.zeros(X.shape[0]) + self.intercept
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])


gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gbr.fit(X, y)

y_pred = gbr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print('MSE:', mse)
print('R^2:', r2)


# Experiment with different hyperparameters such as learning rate, number of trees, and tree depth to
# optimise the performance of the model. Use grid search or random search to find the best
# hyperparameters

# In[ ]:



from sklearn.model_selection import GridSearchCV

gbr = GradientBoostingRegressor()

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [2, 3, 4]
}

grid_search = GridSearchCV(gbr, param_grid=param_grid, cv=5)
grid_search.fit(X, y)

print('Best hyperparameters:', grid_search.best_params_)

best_gbr = GradientBoostingRegressor(**grid_search.best_params_)
best_gbr.fit(X, y)

y_pred = best_gbr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print('MSE:', mse)
print('R^2:', r2)


# What is a weak learner in Gradient Boosting?

# In Gradient Boosting, a weak learner is a model that performs slightly better than random guessing. Typically, weak learners are decision trees with a small depth (e.g. 1 or 2) or linear models with a small number of features.
# 
# The idea behind using weak learners in Gradient Boosting is that they can be combined to form a strong learner that can make accurate predictions. Each weak learner focuses on a different aspect of the data, and the errors made by one weak learner can be corrected by subsequent weak learners in the boosting process.
# 
# The term "weak" does not necessarily mean that the model is of low quality, but rather that it has limited capacity to fit the data on its own. The strength of the model comes from combining multiple weak learners into an ensemble.

# What is the intuition behind the Gradient Boosting algorithm?

# The intuition behind the Gradient Boosting algorithm is to iteratively train a sequence of weak models (e.g. decision trees) and combine them to create a strong model that can accurately predict the target variable.
# 
# At each iteration of the algorithm, the model attempts to fit the residual errors from the previous iteration. In other words, the model learns to predict the difference between the true target values and the predicted target values from the previous iteration. This process is repeated for a fixed number of iterations or until a stopping criterion is met.
# 
# By focusing on the errors made by the previous model, the Gradient Boosting algorithm places more emphasis on the examples that were poorly predicted by the previous model. This allows the algorithm to learn from its mistakes and improve its predictions over time.
# 
# To combine the weak models into a strong model, Gradient Boosting uses a weighted sum of the individual model predictions. The weights assigned to each model are determined during training, based on how well each model performs on the training set.

# How does Gradient Boosting algorithm build an ensemble of weak learners?

# Initialize the model: The first step is to initialize the model by fitting a weak learner to the training data. This could be a simple model such as a decision tree with a small number of nodes.
# 
# Compute the residuals: Next, the model computes the difference between the predicted target values and the true target values for each training example. These differences are known as residuals.
# 
# Fit a weak learner to the residuals: The next step is to fit another weak learner to the residuals computed in the previous step. This model is designed to predict the residual errors made by the previous model.
# 
# Update the model: The model updates its predictions by adding the predictions of the new weak learner to the predictions of the previous model. This produces a new set of predictions that are hopefully more accurate than the previous ones.
# 
# Repeat: The previous steps are repeated until a stopping criterion is met. This could be a fixed number of iterations or until the model achieves a certain level of performance.
# 
# Combine the weak learners: Finally, the weak learners are combined into a strong ensemble model by taking a weighted sum of their predictions. The weights are determined based on the performance of each weak learner on the training set.
# 
# 

# What are the steps involved in constructing the mathematical intuition of Gradient Boosting
# algorithm?

# Define the loss function: The first step is to define a loss function that measures the difference between the predicted target values and the true target values. This loss function should be differentiable, so that it can be optimized using gradient descent.
# 
# Initialize the model: The second step is to initialize the model by fitting a weak learner to the training data. This could be a simple model such as a decision tree with a small number of nodes.
# 
# Compute the negative gradient of the loss function: The next step is to compute the negative gradient of the loss function with respect to the predicted target values. This represents the direction in which the loss function is decreasing the most.
# 
# Fit a weak learner to the negative gradient: The next step is to fit another weak learner to the negative gradient computed in the previous step. This model is designed to predict the negative gradient of the loss function.
# 
# Update the model: The model updates its predictions by adding the predictions of the new weak learner to the predictions of the previous model, weighted by a learning rate parameter. This produces a new set of predictions that are hopefully more accurate than the previous ones.
# 
# Repeat: The previous steps are repeated until a stopping criterion is met. This could be a fixed number of iterations or until the model achieves a certain level of performance.
# 
# Combine the weak learners: Finally, the weak learners are combined into a strong ensemble model by taking a weighted sum of their predictions. The weights are determined based on the performance of each weak learner on the training set.

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

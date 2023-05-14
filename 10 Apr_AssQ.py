#!/usr/bin/env python
# coding: utf-8

# A company conducted a survey of its employees and found that 70% of the employees use the
# company's health insurance plan, while 40% of the employees who use the plan are smokers. What is the
# probability that an employee is a smoker given that he/she uses the health insurance plan?

# P(smoker | uses plan) = P(uses plan | smoker) * P(smoker) / P(uses plan)
# 
# We are given that 70% of the employees use the health insurance plan, so P(uses plan) = 0.7. We are also given that 40% of the employees who use the plan are smokers, so P(uses plan | smoker) = 0.4. Finally, we are not given the prior probability of being a smoker, but we can calculate it from the information given:
# 
# P(smoker) = P(uses plan and smoker) + P(uses plan and non-smoker)
# = P(uses plan | smoker) * P(smoker) + P(uses plan | non-smoker) * P(non-smoker)
# = 0.4 * P(smoker) + P(uses plan | non-smoker) * (1 - P(smoker))
# 
# We don't know the value of P(uses plan | non-smoker), but we can use the fact that only 70% of employees use the health insurance plan to write:
# 
# P(uses plan | non-smoker) = P(uses plan and non-smoker) / P(non-smoker)
# = (1 - P(smoker)) * 0.3 / (1 - P(smoker))
# 
# Substituting this into the equation above and solving for P(smoker), we get:
# 
# P(smoker) = 0.1333
# 
# Substituting all the values we have into the original equation, we get:
# 
# P(smoker | uses plan) = 0.4 * 0.1333 / 0.7
# 
# Simplifying, we get:
# 
# P(smoker | uses plan) = 0.076
# 
# Therefore, the probability that an employee is a smoker given that he/she uses the health insurance plan is 0.076, or about 7.6%.

# What is the difference between Bernoulli Naive Bayes and Multinomial Naive Bayes?

# Bernoulli Naive Bayes assumes that the features are binary and calculates the probability of each feature being present or absent in a given class. This makes it useful for problems where the features are simple presence/absence indicators, such as spam filtering.
# 
# Multinomial Naive Bayes assumes that the features are counts of discrete events (such as word frequencies), and calculates the probability of each feature occurring in a given class. This makes it useful for text classification problems, where the input data is typically represented as a bag of words.
# 
# In Bernoulli Naive Bayes, each feature is considered independent of all other features, which is why the algorithm is called "naive". In Multinomial Naive Bayes, the features are assumed to be conditionally independent given the class label.
# Bernoulli Naive Bayes is often used for problems with small datasets or where the feature space is very large, because it is computationally efficient and requires relatively few parameters to be estimated. Multinomial Naive Bayes is often used for text classification problems with large datasets, because it can handle sparse data and is well-suited for problems where the number of features is very large.

# How does Bernoulli Naive Bayes handle missing values?
# 

# In Bernoulli Naive Bayes, missing values are typically handled by assuming that the feature is not present. This is based on the assumption that each feature is binary, and so a missing value can be treated as a "0" value (i.e., the feature is not present) without affecting the underlying probability distribution.
# 
# Specifically, in Bernoulli Naive Bayes, each feature is represented by a binary variable indicating whether the feature is present (1) or absent (0). If a data point has a missing value for a particular feature, the algorithm assumes that the feature is absent (i.e., it is represented as a 0). This is also known as the "missing not at random" (MNAR) assumption, where the probability of a feature being missing depends on the feature itself, and not just on the other observed features.

# Can Gaussian Naive Bayes be used for multi-class classification?

# Yes, Gaussian Naive Bayes can be used for multi-class classification.
# 
# In Gaussian Naive Bayes, the algorithm assumes that the features follow a Gaussian (normal) distribution within each class. This means that the probability density function of each feature is assumed to be a normal distribution with a mean and variance specific to each class.
# 
# To extend Gaussian Naive Bayes to multi-class classification, the algorithm can be trained on a dataset with more than two classes by simply estimating the mean and variance of each feature for each class. During prediction, the algorithm calculates the probability of the input belonging to each class using the class-specific mean and variance estimates, and then selects the class with the highest probability as the predicted class.

# Data preparation:
# Download the "Spambase Data Set" from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/
# datasets/Spambase). This dataset contains email messages, where the goal is to predict whether a message
# is spam or not based on several input features.
# Implementation:
# Implement Bernoulli Naive Bayes, Multinomial Naive Bayes, and Gaussian Naive Bayes classifiers using the
# scikit-learn library in Python. Use 10-fold cross-validation to evaluate the performance of each classifier on the
# dataset. You should use the default hyperparameters for each classifier.
# Results:
# Report the following performance metrics for each classifier:
# Accuracy
# Precision
# Recall
# F1 score
# Discussion:
# Discuss the results you obtained. Which variant of Naive Bayes performed the best? Why do you think that is
# the case? Are there any limitations of Naive Bayes that you observed?
# Conclusion:
# Summarise your findings and provide some suggestions for future work.

# In[19]:


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

data = pd.read_csv("C:/Users/dvkha/Downloads/spambase.data", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

bernoulli_clf = BernoulliNB()
multinomial_clf = MultinomialNB()
gaussian_clf = GaussianNB()

bernoulli_scores = cross_val_score(bernoulli_clf, X, y, cv=10)
multinomial_scores = cross_val_score(multinomial_clf, X, y, cv=10)
gaussian_scores = cross_val_score(gaussian_clf, X, y, cv=10)

print("Bernoulli Naive Bayes:")
print("Accuracy:", bernoulli_scores.mean())
print("Precision:", cross_val_score(bernoulli_clf, X, y, cv=10, scoring='precision').mean())
print("Recall:", cross_val_score(bernoulli_clf, X, y, cv=10, scoring='recall').mean())
print("F1 score:", cross_val_score(bernoulli_clf, X, y, cv=10, scoring='f1').mean())

print("Multinomial Naive Bayes:")
print("Accuracy:", multinomial_scores.mean())
print("Precision:", cross_val_score(multinomial_clf, X, y, cv=10, scoring='precision').mean())
print("Recall:", cross_val_score(multinomial_clf, X, y, cv=10, scoring='recall').mean())
print("F1 score:", cross_val_score(multinomial_clf, X, y, cv=10, scoring='f1').mean())

print("Gaussian Naive Bayes:")
print("Accuracy:", gaussian_scores.mean())
print("Precision:", cross_val_score(gaussian_clf, X, y, cv=10, scoring='precision').mean())
print("Recall:", cross_val_score(gaussian_clf, X, y, cv=10, scoring='recall').mean())
print("F1 score:", cross_val_score(gaussian_clf, X, y, cv=10, scoring='f1').mean())


# In[ ]:





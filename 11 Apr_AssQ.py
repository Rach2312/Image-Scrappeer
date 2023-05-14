#!/usr/bin/env python
# coding: utf-8

# What is an ensemble technique in machine learning?

# An ensemble technique in machine learning is a method of combining multiple individual models to improve the overall performance of the system. The idea behind ensemble techniques is that by combining different models that have varying strengths and weaknesses, the resulting ensemble can be more accurate and robust than any individual model.

# Why are ensemble techniques used in machine learning?

# Ensemble techniques are used in machine learning for several reasons:
# 
# Improved accuracy: By combining the predictions of multiple models, ensemble techniques can achieve higher accuracy than any individual model. This is because the ensemble can leverage the strengths of each individual model and compensate for their weaknesses.
# 
# Robustness: Ensemble techniques can be more robust than individual models, as they are less likely to be affected by noise or outliers in the data.
# 
# Overfitting prevention: Ensemble techniques can help prevent overfitting, where a model learns to fit the training data too closely and performs poorly on new data. This is because the ensemble can generalize better by combining the predictions of multiple models that have been trained on different subsets of the data.
# 
# Versatility: Ensemble techniques can be applied to a wide range of machine learning tasks, including classification, regression, and clustering.

# What is bagging?

# Bagging (Bootstrap Aggregation) is an ensemble technique in machine learning that involves training multiple instances of the same model on different subsets of the training data and combining their predictions through averaging or voting. The key idea behind bagging is to reduce the variance of the model by introducing randomness in the training process.

# What is boosting?

# Boosting is an ensemble technique in machine learning that involves training a series of weak models sequentially, with each model trying to correct the errors of its predecessor. The key idea behind boosting is to improve the performance of the model by focusing on the examples that are difficult to classify.

# What are the benefits of using ensemble techniques?

# There are several benefits of using ensemble techniques in machine learning:
# 
# Improved accuracy: Ensemble techniques can achieve higher accuracy than any individual model by combining the predictions of multiple models. This is because the ensemble can leverage the strengths of each individual model and compensate for their weaknesses.
# 
# Robustness: Ensemble techniques can be more robust than individual models, as they are less likely to be affected by noise or outliers in the data.
# 
# Overfitting prevention: Ensemble techniques can help prevent overfitting, where a model learns to fit the training data too closely and performs poorly on new data. This is because the ensemble can generalize better by combining the predictions of multiple models that have been trained on different subsets of the data.
# 
# Versatility: Ensemble techniques can be applied to a wide range of machine learning tasks, including classification, regression, and clustering.

# Are ensemble techniques always better than individual models?

# Ensemble techniques are not always better than individual models, as their performance depends on several factors, including the type of data, the quality of the individual models, and the method of combination.
# 
# In some cases, an individual model may perform better than an ensemble of models, especially if the individual model is well-suited to the data and has been trained on a large and diverse dataset. Additionally, if the individual model is already a complex ensemble of models, further ensemble techniques may not provide significant improvements.

# How is the confidence interval calculated using bootstrap?

# The confidence interval using bootstrap can be calculated using the following steps:
# 
# Collect a sample of size n from the population.
# Create B bootstrap samples by randomly sampling n observations with replacement from the original sample.
# Calculate the sample statistic of interest (e.g., mean, median, standard deviation, etc.) for each of the bootstrap samples.
# Calculate the standard error of the statistic by taking the standard deviation of the B bootstrap sample statistics.
# Calculate the confidence interval for the population parameter by subtracting and adding the appropriate z-score (or t-score if the sample size is small) times the standard error to the sample statistic.

# How does bootstrap work and What are the steps involved in bootstrap?

# The following are the steps involved in bootstrap:
# 
# Collect a sample of size n from the population of interest.
# 
# Randomly select n observations from the sample with replacement to create a new bootstrap sample. Some observations may be selected multiple times, while others may not be selected at all.
# 
# Estimate the statistic of interest (e.g., mean, median, standard deviation, etc.) from the bootstrap sample.
# 
# Repeat steps 2 and 3 B times, creating B bootstrap samples and estimating the statistic from each sample.
# 
# Calculate the bootstrap distribution of the statistic by compiling the B estimates into a single distribution.
# 
# Calculate the standard error of the statistic, which is the standard deviation of the bootstrap distribution.
# 
# Calculate the confidence interval for the population parameter using the bootstrap distribution and the desired level of confidence.

# A researcher wants to estimate the mean height of a population of trees. They measure the height of a
# sample of 50 trees and obtain a mean height of 15 meters and a standard deviation of 2 meters. Use
# bootstrap to estimate the 95% confidence interval for the population mean height.

# In[20]:


import numpy as np


sample_heights = np.array([15.2, 14.9, 15.4, 16.3, 13.9, 14.5, 15.8, 14.2, 15.3, 15.6,
                           15.1, 15.7, 14.8, 14.4, 15.5, 14.6, 15.0, 15.9, 16.2, 13.8,
                           14.3, 14.7, 15.7, 15.2, 16.0, 14.5, 15.4, 15.1, 16.1, 14.2,
                           15.3, 14.9, 15.8, 14.1, 15.5, 15.2, 15.9, 15.6, 14.4, 14.8,
                           15.0, 15.7, 13.7, 14.3, 15.1, 16.2, 14.6, 16.0, 15.3, 14.7])


sample_mean = np.mean(sample_heights)
sample_std = np.std(sample_heights)


B = 10000


bootstrap_means = np.zeros(B)
for i in range(B):
    bootstrap_sample = np.random.choice(sample_heights, size=50, replace=True)
    bootstrap_means[i] = np.mean(bootstrap_sample)


se_bootstrap = np.std(bootstrap_means)


lower_ci = np.percentile(bootstrap_means, 2.5)
upper_ci = np.percentile(bootstrap_means, 97.5)

print("Bootstrap estimate of mean height: {:.2f}".format(np.mean(bootstrap_means)))
print("Bootstrap estimate of standard error: {:.2f}".format(se_bootstrap))
print("95% Confidence interval: [{:.2f}, {:.2f}]".format(lower_ci, upper_ci))


# In[ ]:





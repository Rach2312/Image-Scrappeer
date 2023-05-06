#!/usr/bin/env python
# coding: utf-8

# What is the difference between Ordinal Encoding and Label Encoding? Provide an example of when you
# might choose one over the other.

# Ordinal encoding and label encoding are two techniques commonly used in machine learning to convert categorical data into numerical data, which can be used as input to machine learning models.
# 
# Label encoding refers to the process of assigning a unique numerical value to each category in a categorical variable. For example, suppose we have a categorical variable "color" with three categories: "red," "green," and "blue." Label encoding would assign the values 0, 1, and 2 to these categories, respectively.
# 
# Ordinal encoding is a similar process, but it assigns numerical values to categories based on their order or rank. For example, if we have a categorical variable "size" with the categories "small," "medium," and "large," ordinal encoding would assign the values 0, 1, and 2 to these categories, respectively, based on their order.

# Explain how Target Guided Ordinal Encoding works and provide an example of when you might use it in
# a machine learning project.

# Target Guided Ordinal Encoding is a technique used to encode categorical variables based on the relationship between the variable and the target variable in a supervised learning problem. The idea behind this encoding is to create a monotonic relationship between the encoded values and the target variable, which can help improve the predictive power of the model.
# 
# The steps involved in Target Guided Ordinal Encoding are as follows:
# 
# For each category in the categorical variable, calculate the mean (or median) of the target variable.
# 
# Order the categories based on the mean (or median) value in ascending or descending order.
# 
# Assign an ordinal value to each category based on its order in the sorted list.
# 
# Replace the original categories with their assigned ordinal values.
# 
# For example, suppose we have a categorical variable "occupation" with four categories: "teacher," "engineer," "doctor," and "lawyer." We want to predict the income level of individuals based on their occupation. To perform Target Guided Ordinal Encoding, we would calculate the mean income for each category:
# 
# Occupation	Mean Income
# Teacher	$50,000
# Engineer	$80,000
# Doctor	$120,000
# Lawyer	$100,000
# Based on the mean income, we would order the categories in descending order:
# 
# Doctor
# Lawyer
# Engineer
# Teacher
# Then, we would assign ordinal values to each category based on their order in the sorted list:
# 
# Occupation	Ordinal Value
# Teacher	1
# Engineer	2
# Doctor	3
# Lawyer	4
# Finally, we would replace the original categories with their assigned ordinal values in the dataset.

# Define covariance and explain why it is important in statistical analysis. How is covariance calculated?

# 
# Covariance is a statistical measure that indicates how two variables are related to each other. It measures the degree to which the variables tend to vary together or apart. Specifically, covariance measures the joint variability of two random variables, or how much they change together or in opposite directions.
# 
# Covariance is an important concept in statistical analysis because it provides a way to quantify the relationship between two variables. If two variables have a high covariance, it means that they tend to move in the same direction. Conversely, if they have a low covariance, it means that they tend to move in opposite directions or are not related.
# 
# Covariance is calculated using the following formula:
# 
# cov(X,Y) = (1/n) * Σ[(Xi - Xbar) * (Yi - Ybar)]
# 
# where X and Y are two random variables, Xi and Yi are their respective observations, Xbar and Ybar are their means, and n is the total number of observations.
# 
# The resulting value of covariance can be positive, negative, or zero. A positive value indicates that the variables tend to move in the same direction, while a negative value indicates that they tend to move in opposite directions. A covariance of zero indicates that the variables are not related.

# For a dataset with the following categorical variables: Color (red, green, blue), Size (small, medium,
# large), and Material (wood, metal, plastic), perform label encoding using Python's scikit-learn library.
# Show your code and explain the output.

# In[2]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.DataFrame({
    'Color': ['red', 'green', 'blue', 'green', 'red'],
    'Size': ['small', 'medium', 'large', 'small', 'medium'],
    'Material': ['wood', 'metal', 'plastic', 'plastic', 'wood']
})


le = LabelEncoder()


data['Color_encoded'] = le.fit_transform(data['Color'])
data['Size_encoded'] = le.fit_transform(data['Size'])
data['Material_encoded'] = le.fit_transform(data['Material'])


print(data)


# Calculate the covariance matrix for the following variables in a dataset: Age, Income, and Education
# level. Interpret the results.

# Assuming we have such a dataset, we can calculate the covariance matrix as follows:
# 
# Calculate the mean (average) of each variable.
# For each pair of variables, calculate the product of the deviations of the observations from their respective means. For example, for Age and Income, we would calculate (age1 - mean_age) * (income1 - mean_income), (age2 - mean_age) * (income2 - mean_income), and so on for each observation in the dataset.
# Sum the products of the deviations for each pair of variables.
# Divide each sum by the number of observations minus one to obtain the sample covariance between the two variables.
# Repeat steps 2-4 for each pair of variables to obtain the covariance matrix.
# The resulting covariance matrix would be a 3x3 matrix with the variances of each variable along the diagonal and the covariances between each pair of variables in the off-diagonal elements. For example, the covariance matrix might look like:
# 
# Age	Income	Education
# Age	50.0	1000.0	-5.0
# Income	1000.0	100000.0	200.0
# Education	-5.0	200.0	1.0
# Interpreting the results of the covariance matrix:
# 
# The diagonal elements represent the variance of each variable. For example, the variance of Age is 50.0. Variance measures how spread out the values of a variable are, so a larger variance indicates that the values of the variable are more spread out from the mean.
# The off-diagonal elements represent the covariance between each pair of variables. For example, the covariance between Age and Income is 1000.0. Covariance measures how two variables vary together, so a positive covariance indicates that the two variables tend to increase or decrease together, while a negative covariance indicates that they tend to vary in opposite directions. In this example, Age and Income have a positive covariance, which could mean that older people tend to have higher incomes, or that people with higher incomes tend to be older.

# You are working on a machine learning project with a dataset containing several categorical
# variables, including "Gender" (Male/Female), "Education Level" (High School/Bachelor's/Master's/PhD),
# and "Employment Status" (Unemployed/Part-Time/Full-Time). Which encoding method would you use for
# each variable, and why?

# Gender: Binary encoding
# Since there are only two categories (Male/Female), binary encoding is a simple and efficient way to encode this variable. We can use a binary code, such as 0/1, to represent the two categories.
# 
# Education Level: Ordinal encoding
# Education level is an ordinal variable, meaning that the categories have a natural order or hierarchy (High School < Bachelor's < Master's < PhD). Ordinal encoding assigns a unique numerical value to each category, based on their order. For example, we could assign the values 1, 2, 3, and 4 to High School, Bachelor's, Master's, and PhD, respectively.
# 
# Employment Status: One-hot encoding
# Employment status is a nominal variable, meaning that the categories do not have a natural order or hierarchy. One-hot encoding creates a new binary feature for each category, and assigns a value of 1 if the category is present and 0 if it is not. For example, we could create three new features: "Unemployed" (1 if the person is unemployed, 0 otherwise), "Part-Time" (1 if the person is working part-time, 0 otherwise), and "Full-Time" (1 if the person is working full-time, 0 otherwise).

# You are analyzing a dataset with two continuous variables, "Temperature" and "Humidity", and two
# categorical variables, "Weather Condition" (Sunny/Cloudy/Rainy) and "Wind Direction" (North/South/
# East/West). Calculate the covariance between each pair of variables and interpret the results.

# In[4]:


import numpy as np


temperature = [20, 25, 30, 22, 27]
humidity = [50, 70, 80, 60, 75]
weather_condition = ['Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Rainy']
wind_direction = ['North', 'South', 'East', 'West', 'North']


continuous_data = np.array([temperature, humidity])


covariance_matrix = np.cov(continuous_data)

print("Covariance Matrix:")
print(covariance_matrix)


# In[ ]:





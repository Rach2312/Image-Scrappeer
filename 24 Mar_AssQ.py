#!/usr/bin/env python
# coding: utf-8

# What are the key features of the wine quality data set? Discuss the importance of each feature in
# predicting the quality of wine.

# The wine quality data set contains information on various physicochemical properties of different types of Portuguese wines, along with their sensory quality rating as determined by human tasters. The dataset has a total of 12 features, including 11 input variables and 1 output variable. The input variables are:
# 
# Fixed acidity: the amount of fixed acids in the wine, which can affect its taste and pH level.
# Volatile acidity: the amount of volatile acids in the wine, which can affect its taste and aroma.
# Citric acid: the amount of citric acid in the wine, which can affect its acidity and freshness.
# Residual sugar: the amount of residual sugar in the wine, which can affect its sweetness and perceived body.
# Chlorides: the amount of salt in the wine, which can affect its taste and stability.
# Free sulfur dioxide: the amount of free sulfur dioxide in the wine, which can affect its antioxidant properties and stability.
# Total sulfur dioxide: the total amount of sulfur dioxide in the wine, which can affect its antioxidant properties and stability.
# Density: the density of the wine, which can affect its perceived body and alcohol content.
# pH: the pH level of the wine, which can affect its taste, stability, and color.
# Sulphates: the amount of sulphates in the wine, which can affect its antioxidant properties and stability.
# Alcohol: the percentage of alcohol in the wine, which can affect its perceived body and taste.

# How did you handle missing data in the wine quality data set during the feature engineering process?
# Discuss the advantages and disadvantages of different imputation techniques.

# One common method for handling missing data is to remove all data points with missing values. This is known as listwise deletion or complete case analysis. The advantage of this method is that it is straightforward and easy to implement. However, it can result in a loss of a large amount of data, which can reduce the sample size and potentially bias the results if the missing data is not missing completely at random.
# 
# Another method is to replace missing values with the mean or median of the non-missing values for that feature. This is known as mean or median imputation. The advantage of this method is that it is also straightforward and can work well when the missing data is missing completely at random. However, it can lead to biased estimates of the true mean or median if the missing data is not missing completely at random. Additionally, it does not account for any relationships between the missing feature and other variables in the dataset.
# 
# In general, the best method for handling missing data depends on the specific dataset and the extent and pattern of the missing data. It is important to carefully consider the advantages and disadvantages of each method and to compare the results obtained using different methods to ensure that the missing data is handled appropriately and does not bias the results of the analysis.

# What are the key factors that affect students' performance in exams? How would you go about
# analyzing these factors using statistical techniques?

# There are many factors that can affect students' performance in exams, some of which include:
# 
# Prior knowledge and ability: students' previous academic performance and ability can influence their performance in exams.
# 
# Learning environment: factors such as classroom setting, teaching style, and class size can all affect how well students perform in exams.
# 
# Study habits: the amount and quality of studying that students do can have a significant impact on their performance.
# 
# Motivation and engagement: students who are motivated and engaged in their studies tend to perform better than those who are not.
# 
# Stress and anxiety: high levels of stress and anxiety can negatively impact students' exam performance.
# 
# To analyze these factors using statistical techniques, one approach would be to collect data on each of these factors and students' exam performance, and then conduct a multiple regression analysis to determine which factors have the strongest relationship with exam performance. This could involve using a statistical software program to run the regression analysis and interpreting the results, which might include coefficients, p-values, and R-squared values.

# Describe the process of feature engineering in the context of the student performance data set. How
# did you select and transform the variables for your model?

# Feature engineering is the process of selecting and transforming variables to create new features that can improve the predictive power of a model. In the context of the student performance data set, the process of feature engineering might involve selecting and transforming variables to better understand the factors that influence student performance and to create new features that can improve the accuracy of a predictive model.
# 
# Here are some steps that might be involved in the feature engineering process for the student performance data set:
# 
# Exploratory data analysis: Before selecting and transforming variables, it's important to conduct exploratory data analysis to get a sense of the distribution of each variable, as well as any relationships among variables. This might involve creating visualizations such as scatterplots, histograms, and boxplots to identify potential outliers, skewness, or patterns in the data.
# 
# Variable selection: Once you have a good understanding of the data, you can select variables that are likely to be important predictors of student performance. This might involve using domain knowledge, such as the factors discussed in the previous answer, or using techniques such as correlation analysis or feature importance rankings to identify the most relevant variables.
# 
# Variable transformation: After selecting the variables, you might want to transform them to better capture the relationship between the variables and student performance. For example, you might transform categorical variables using one-hot encoding, or create new features by combining multiple variables or adding polynomial or interaction terms. Additionally, you might want to standardize or normalize continuous variables to ensure that they have the same scale and to improve the performance of certain machine learning algorithms.
# 
# Feature selection: Finally, you might want to use techniques such as regularization or feature importance rankings to select the most relevant features for your model. This can help reduce the risk of overfitting and improve the interpretability of the model.

# Load the wine quality data set and perform exploratory data analysis (EDA) to identify the distribution
# of each feature. Which feature(s) exhibit non-normality, and what transformations could be applied to
# these features to improve normality?

# In[7]:


pip install seaborn


# In[6]:


pip install pandas


# In[9]:


import pandas as pd
import seaborn as sns


wine = pd.read_csv('winequality-red.csv')


sns.displot(wine['fixed acidity'], kde=True)
sns.displot(wine['volatile acidity'], kde=True)
sns.displot(wine['citric acid'], kde=True)
sns.displot(wine['residual sugar'], kde=True)
sns.displot(wine['chlorides'], kde=True)
sns.displot(wine['free sulfur dioxide'], kde=True)
sns.displot(wine['total sulfur dioxide'], kde=True)
sns.displot(wine['density'], kde=True)
sns.displot(wine['pH'], kde=True)
sns.displot(wine['sulphates'], kde=True)
sns.displot(wine['alcohol'], kde=True)
sns.displot(wine['quality'], kde=True)


# Using the wine quality data set, perform principal component analysis (PCA) to reduce the number of
# features. What is the minimum number of principal components required to explain 90% of the variance in
# the data?

# In[10]:


import pandas as pd
from sklearn.preprocessing import StandardScaler


wine = pd.read_csv('winequality-red.csv')


X = wine.drop('quality', axis=1)
y = wine['quality']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[11]:


from sklearn.decomposition import PCA


pca = PCA()
pca.fit(X_scaled)


variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(variance_ratio)
print(cumulative_variance_ratio)


# 

# 

# 

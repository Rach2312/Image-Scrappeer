#!/usr/bin/env python
# coding: utf-8

# What is Bayes' theorem?

# Bayes' theorem is a mathematical formula that describes the probability of an event, based on prior knowledge of related conditions that might be relevant to the event. The theorem is named after Reverend Thomas Bayes, an 18th-century statistician and philosopher who first formulated the idea.
# 
# The theorem states that the probability of a hypothesis (H) being true given evidence (E) is proportional to the probability of the evidence given the hypothesis, multiplied by the prior probability of the hypothesis, divided by the marginal probability of the evidence. 

# What is the formula for Bayes' theorem?

# Mathematically, this can be represented as:
# 
# P(H | E) = P(E | H) * P(H) / P(E)
# 
# where:
# 
# P(H | E) is the posterior probability of the hypothesis given the evidence
# P(E | H) is the likelihood of the evidence given the hypothesis
# P(H) is the prior probability of the hypothesis
# P(E) is the marginal probability of the evidence

# How is Bayes' theorem used in practice?

# Some practical applications of Bayes' theorem include:
# 
# Spam filtering: Bayes' theorem can be used to classify emails as spam or not spam. The algorithm calculates the probability that an email is spam based on the words it contains and compares it to a threshold. If the probability is higher than the threshold, the email is classified as spam.
# 
# Medical diagnosis: Bayes' theorem can be used to diagnose diseases based on symptoms. The algorithm calculates the probability that a patient has a certain disease based on their symptoms and compares it to a threshold. If the probability is higher than the threshold, the patient is diagnosed with the disease.
# 
# Sentiment analysis: Bayes' theorem can be used to classify text as positive or negative. The algorithm calculates the probability that a piece of text expresses a positive sentiment based on the words it contains and compares it to a threshold. If the probability is higher than the threshold, the text is classified as positive.
# 
# Fraud detection: Bayes' theorem can be used to detect fraudulent transactions. The algorithm calculates the probability that a transaction is fraudulent based on the transaction history and compares it to a threshold. If the probability is higher than the threshold, the transaction is flagged as fraudulent.
# 
# 

# What is the relationship between Bayes' theorem and conditional probability?
# 

# Bayes' theorem and conditional probability are closely related concepts. Conditional probability is the probability of an event occurring given that another event has already occurred. Bayes' theorem provides a way to update the probability of an event based on new information.
# 
# Mathematically, the relationship between Bayes' theorem and conditional probability can be expressed as follows:
# 
# P(A | B) = P(B | A) * P(A) / P(B)
# 
# where:
# 
# P(A | B) is the conditional probability of event A given event B
# P(B | A) is the conditional probability of event B given event A
# P(A) is the prior probability of event A
# P(B) is the marginal probability of event B

# How do you choose which type of Naive Bayes classifier to use for any given problem?

# some guidelines for selecting the appropriate type of Naive Bayes classifier:
# 
# Gaussian Naive Bayes: This classifier is suitable for continuous data that can be modeled using a normal distribution. It is commonly used in classification problems involving real-valued features, such as in medical diagnosis or spam filtering.
# 
# Multinomial Naive Bayes: This classifier is suitable for discrete data that can be represented as counts or frequencies, such as text data. It is commonly used in classification problems involving text classification or sentiment analysis.
# 
# Bernoulli Naive Bayes: This classifier is similar to Multinomial Naive Bayes but is designed for binary or boolean features, such as the presence or absence of a particular word in a document. It is commonly used in classification problems involving binary or boolean data, such as in spam filtering or fraud detection.

# You have a dataset with two features, X1 and X2, and two possible classes, A and B. You want to use Naive
# Bayes to classify a new instance with features X1 = 3 and X2 = 4. The following table shows the frequency of
# each feature value for each class:
# Class X1=1 X1=2 X1=3 X2=1 X2=2 X2=3 X2=4
# A 3 3 4 4 3 3 3
# B 2 2 1 2 2 2 3
# Assuming equal prior probabilities for each class, which class would Naive Bayes predict the new instance
# to belong to?

# Using Bayes' theorem, the posterior probability of class A given the features can be calculated as follows:
# 
# P(A | X1=3, X2=4) = P(X1=3, X2=4 | A) * P(A) / P(X1=3, X2=4)
# 
# Similarly, the posterior probability of class B given the features can be calculated as:
# 
# P(B | X1=3, X2=4) = P(X1=3, X2=4 | B) * P(B) / P(X1=3, X2=4)
# 
# Assuming equal prior probabilities for each class, we can simplify the equations to:
# 
# P(A | X1=3, X2=4) = P(X1=3, X2=4 | A) / P(X1=3, X2=4)
# 
# P(B | X1=3, X2=4) = P(X1=3, X2=4 | B) / P(X1=3, X2=4)
# 
# To calculate the probabilities, we can use the frequency table provided in the problem:
# 
# P(X1=3, X2=4 | A) = 4/16 * 3/16 = 0.046875
# 
# P(X1=3, X2=4 | B) = 1/12 * 3/12 = 0.020833
# 
# P(X1=3, X2=4) = P(X1=3, X2=4 | A) * P(A) + P(X1=3, X2=4 | B) * P(B)
# = 0.046875 * 0.5 + 0.020833 * 0.5
# = 0.033854
# 
# Therefore, the posterior probabilities can be calculated as follows:
# 
# P(A | X1=3, X2=4) = 0.046875 / 0.033854 = 1.386
# P(B | X1=3, X2=4) = 0.020833 / 0.033854 = 0.614
# 
# Since P(A | X1=3, X2=4) > P(B | X1=3, X2=4), Naive Bayes would predict that the new instance belongs to class A.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





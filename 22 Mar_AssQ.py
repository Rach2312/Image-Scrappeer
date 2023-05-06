#!/usr/bin/env python
# coding: utf-8

# Pearson correlation coefficient is a measure of the linear relationship between two variables. Suppose
# you have collected data on the amount of time students spend studying for an exam and their final exam
# scores. Calculate the Pearson correlation coefficient between these two variables and interpret the result.

# Pearson correlation coefficient:
# 
# r = Cov(X,Y) / (SD(X) * SD(Y))
# 
# where r is the Pearson correlation coefficient, Cov(X,Y) is the covariance between X and Y, and SD(X) and SD(Y) are the standard deviations of X and Y, respectively.
# 
# Assuming you have collected data on study time and exam scores for n students, you can calculate the Pearson correlation coefficient as follows:
# 
# Compute the mean study time and the mean exam score:
# 
# x̄ = (Σx) / n and ȳ = (Σy) / n
# 
# Compute the covariance between study time and exam score:
# 
# Cov(X,Y) = [Σ(x - x̄)(y - ȳ)] / (n - 1)
# 
# Compute the standard deviation of study time and exam score:
# 
# SD(X) = sqrt{ [Σ(x - x̄)^2] / (n - 1) } and SD(Y) = sqrt{ [Σ(y - ȳ)^2] / (n - 1) }
# 
# Finally, plug these values into the formula for r:
# 
# r = Cov(X,Y) / (SD(X) * SD(Y))
# 
# Interpreting the result:
# 
# The Pearson correlation coefficient ranges from -1 to 1, where a value of -1 indicates a perfect negative linear relationship, 0 indicates no linear relationship, and 1 indicates a perfect positive linear relationship.
# 
# If the calculated Pearson correlation coefficient is positive, it means that there is a positive linear relationship between study time and exam scores, which indicates that as the amount of time students spend studying increases, their exam scores tend to increase as well. If the calculated Pearson correlation coefficient is negative, it means that there is a negative linear relationship between study time and exam scores, which indicates that as the amount of time students spend studying increases, their exam scores tend to decrease.
# 
# If the calculated Pearson correlation coefficient is close to 0, it means that there is no linear relationship between study time and exam scores, and the amount of time students spend studying does not appear to be related to their exam scores.

# Spearman's rank correlation is a measure of the monotonic relationship between two variables.
# Suppose you have collected data on the amount of sleep individuals get each night and their overall job
# satisfaction level on a scale of 1 to 10. Calculate the Spearman's rank correlation between these two
# variables and interpret the result.}

# Spearman's rank correlation coefficient:
# 
# ρ = 1 - (6 * Σd^2) / (n * (n^2 - 1))
# 
# where ρ is the Spearman's rank correlation coefficient, d is the difference between the ranks of each observation for each variable, and n is the number of observations.
# 
# Assuming you have collected data on sleep and job satisfaction levels for n individuals, you can calculate the Spearman's rank correlation coefficient as follows:
# 
# Convert the values of each variable into ranks, from lowest to highest. If there are ties, assign the average rank to all tied observations.
# 
# Calculate the difference between the ranks of each observation for each variable.
# 
# Compute the Spearman's rank correlation coefficient using the formula:
# 
# ρ = 1 - (6 * Σd^2) / (n * (n^2 - 1))
# 
# Interpreting the result:
# 
# The Spearman's rank correlation coefficient ranges from -1 to 1, where a value of -1 indicates a perfect negative monotonic relationship, 0 indicates no monotonic relationship, and 1 indicates a perfect positive monotonic relationship.
# 
# If the calculated Spearman's rank correlation coefficient is positive, it means that there is a positive monotonic relationship between the amount of sleep individuals get each night and their overall job satisfaction level. This indicates that as the amount of sleep individuals get each night increases, their job satisfaction level tends to increase as well, in a monotonic fashion. If the calculated Spearman's rank correlation coefficient is negative, it means that there is a negative monotonic relationship between the two variables, which indicates that as the amount of sleep individuals get each night increases, their job satisfaction level tends to decrease in a monotonic fashion.
# 
# If the calculated Spearman's rank correlation coefficient is close to 0, it means that there is no monotonic relationship between the amount of sleep individuals get each night and their overall job satisfaction level, and the amount of sleep individuals get each night does not appear to be related to their job satisfaction level in a monotonic fashion.

# Suppose you are conducting a study to examine the relationship between the number of hours of
# exercise per week and body mass index (BMI) in a sample of adults. You collected data on both variables
# for 50 participants. Calculate the Pearson correlation coefficient and the Spearman's rank correlation
# between these two variables and compare the results.}

# Assuming you have collected data on exercise hours and BMI for 50 participants, you can calculate the Pearson correlation coefficient and the Spearman's rank correlation coefficient as follows:
# 
# Calculate the mean exercise hours and the mean BMI:
# 
# x̄ = (Σx) / n and ȳ = (Σy) / n
# 
# Calculate the covariance between exercise hours and BMI:
# 
# Cov(X,Y) = [Σ(x - x̄)(y - ȳ)] / (n - 1)
# 
# Calculate the standard deviation of exercise hours and BMI:
# 
# SD(X) = sqrt{ [Σ(x - x̄)^2] / (n - 1) } and SD(Y) = sqrt{ [Σ(y - ȳ)^2] / (n - 1) }
# 
# Calculate the Pearson correlation coefficient using the formula:
# 
# r = Cov(X,Y) / (SD(X) * SD(Y))
# 
# To calculate the Spearman's rank correlation coefficient, first convert the values of each variable into ranks, from lowest to highest. If there are ties, assign the average rank to all tied observations.
# 
# Calculate the difference between the ranks of each observation for each variable.
# 
# Calculate the Spearman's rank correlation coefficient using the formula:
# 
# ρ = 1 - (6 * Σd^2) / (n * (n^2 - 1))
# 
# where ρ is the Spearman's rank correlation coefficient, d is the difference between the ranks of each observation for each variable, and n is the number of observations.
# 
# After calculating both coefficients, you can compare the results.
# 
# If the Pearson correlation coefficient is significantly different from zero, it means that there is a significant linear relationship between the two variables. On the other hand, if the Spearman's rank correlation coefficient is significantly different from zero, it means that there is a significant monotonic relationship between the two variables, but not necessarily a linear one.

# A researcher is interested in examining the relationship between the number of hours individuals
# spend watching television per day and their level of physical activity. The researcher collected data on
# both variables from a sample of 50 participants. Calculate the Pearson correlation coefficient between
# these two variables.

# To calculate the Pearson correlation coefficient between the number of hours individuals spend watching television per day and their level of physical activity, you can follow these steps:
# 
# Collect data on both variables from a sample of 50 participants.
# 
# Calculate the mean number of hours spent watching television per day (x̄) and the mean level of physical activity (ȳ).
# 
# Calculate the deviation of each participant's number of hours spent watching television per day (x) from the mean number of hours (x - x̄).
# 
# Calculate the deviation of each participant's level of physical activity (y) from the mean level of physical activity (y - ȳ).
# 
# Multiply each participant's x deviation by their y deviation to get x * y for each participant.
# 
# Sum up the x deviations, y deviations, and x * y for all participants.
# 
# Calculate the sample standard deviation of the number of hours spent watching television per day (SD(x)) and the sample standard deviation of the level of physical activity (SD(y)).
# 
# Calculate the Pearson correlation coefficient using the formula:
# 
# r = Σ[(x - x̄) * (y - ȳ)] / (SD(x) * SD(y) * (n - 1))
# 
# where r is the Pearson correlation coefficient, x̄ is the mean number of hours spent watching television per day, ȳ is the mean level of physical activity, SD(x) is the sample standard deviation of the number of hours spent watching television per day, SD(y) is the sample standard deviation of the level of physical activity, and n is the sample size.
# 
# After following these steps, you should have calculated the Pearson correlation coefficient between the number of hours individuals spend watching television per day and their level of physical activity.
# 
# The resulting correlation coefficient (r) will be a number between -1 and 1, with -1 indicating a perfect negative correlation (as one variable increases, the other decreases), 1 indicating a perfect positive correlation (as one variable increases, the other increases), and 0 indicating no correlation.
# 
# Interpreting the coefficient in the context of the study, a negative coefficient would suggest that as the number of hours individuals spend watching television per day increases, their level of physical activity decreases. A positive coefficient would suggest that as the number of hours individuals spend watching television per day increases, their level of physical activity also increases. A coefficient close to 0 would suggest that there is no relationship between the two variables.

# A survey was conducted to examine the relationship between age and preference for a particular
# brand of soft drink. The survey results are shown below:
# 
# Age(Years)     Soft drink Preference
# 25 Coke        
# 42 Pepsi
# 37  Mountain dew
# 19 Coke 
# 31 Pepsi
# 28 Coke   

# Create a contingency table of the data, with age (in years) in one column and soft drink preference in another column:
# Age (Years)	Soft drink preference
# 25	Coke
# 42	Pepsi
# 37	Mountain Dew
# 19	Coke
# 31	Pepsi
# 28	Coke
# Calculate the expected frequencies for each cell of the contingency table. To do this, first calculate the row and column totals:
# Age (Years)	Soft drink preference	Row total
# 25	Coke	1
# 42	Pepsi	1
# 37	Mountain Dew	1
# 19	Coke	1
# 31	Pepsi	1
# 28	Coke	1
# Column total		6
# The expected frequency for each cell is calculated as:
# 
# Expected frequency = (row total * column total) / sample size
# 
# So, for example, the expected frequency for the cell where age is 25 and soft drink preference is Coke is:
# 
# Expected frequency = (1 * 3) / 6 = 0.5
# 
# Calculate the expected frequencies for all cells:
# 
# Age (Years)	Soft drink preference	Observed frequency	Expected frequency
# 25	Coke	1	0.5
# 42	Pepsi	1	0.5
# 37	Mountain Dew	1	0.5
# 19	Coke	1	0.5
# 31	Pepsi	1	0.5
# 28	Coke	1	0.5
# Column total		6	
# Calculate the chi-square test statistic using the formula:
# chi-square = Σ[(O - E)^2 / E]
# 
# where O is the observed frequency and E is the expected frequency for each cell.
# 
# For example, the contribution to the chi-square value from the first cell (age 25 and Coke preference) is:
# 
# (1 - 0.5)^2 / 0.5 = 0.5
# 
# Calculate the contribution to the chi-square value for all cells and sum them up:
# 
# chi-square = 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 = 3
# 
# Calculate the degrees of freedom for the test. For a contingency table with r rows and c columns, the degrees of freedom is (r - 1) * (c - 1). In this case, there are 2 rows and 3 columns, so the degrees of freedom is 2.
# 
# Look up the critical value of the chi-square distribution for the desired level of significance and degrees of freedom. For example, at a significance level of 0.05 and 2 degrees of freedom, the critical value is 5.99.
# 
# Compare the calculated chi-square value to the critical value. If the calculated value is greater than the critical value, reject the null hypothesis that there is no association

# A company is interested in examining the relationship between the number of sales calls made per day
# and the number of sales made per week. The company collected data on both variables from a sample of
# 30 sales representatives. Calculate the Pearson correlation coefficient between these two variables.

# Organize the data in two columns, one for the number of sales calls made per day and one for the number of sales made per week:
# Sales Calls per Day	Sales per Week
# 10	15
# 8	12
# 11	16
# 12	17
# 9	13
# 7	10
# 10	15
# 13	20
# 11	16
# 9	14
# 8	12
# 10	15
# 12	18
# 13	19
# 11	17
# 9	14
# 7	11
# 10	15
# 12	18
# 13	20
# 11	17
# 9	13
# 8	12
# 10	15
# 12	19
# 13	20
# 11	16
# 9	14
# 7	10
# 10	15
# 12	18
# Calculate the mean and standard deviation for both variables:
# Sales Calls per Day	Sales per Week
# Mean	10.1	15.6
# Standard Deviation	1.7	2.5
# Calculate the covariance between the two variables using the formula:
# cov(X,Y) = Σ[(Xi - X_mean) * (Yi - Y_mean)] / (n - 1)
# 
# where X is the variable of sales calls per day, Y is the variable of sales made per week, n is the sample size, X_mean is the mean of X, and Y_mean is the mean of Y.
# 
# In this case:
# 
# cov(X,Y) = Σ[(Xi - 10.1) * (Yi - 15.6)] / 29
# 
# After calculating for all pairs of X and Y values, we get:
# 
# cov(X,Y) = 15.41
# 
# Calculate the standard deviation of both variables and multiply them together:
# sXsY = sX * sY = 1.7 * 2.5 = 4.25
# 
# Calculate the Pearson correlation coefficient using the formula:
# r = cov(X,Y) / (sX * sY)
# 
# In this case:
# 
# r = 15.41 / 4.25 = 3.63
# 
# The Pearson correlation coefficient between the number of sales calls made per day and the number of sales made per week for this sample of 30 sales representatives is 0.63.
# 
# Interpretation: The Pearson correlation coefficient is a measure of the strength of the linear relationship between two variables, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation). A correlation coefficient of 0 indicates no linear relationship between the variables. In this case, a correlation coefficient of 0.63 suggests a moderately positive correlation between the number

# 

# 

# 

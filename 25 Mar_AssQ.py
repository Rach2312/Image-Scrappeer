#!/usr/bin/env python
# coding: utf-8

# Load the flight price dataset and examine its dimensions. How many rows and columns does the
# dataset have?

# In[12]:


import pandas as pd


df = pd.read_csv('flight_price.xlsx - Sheet1.csv')


print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# What is the distribution of flight prices in the dataset? Create a histogram to visualize the
# distribution.

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('flight_price.xlsx - Sheet1.csv')


plt.hist(df['Price'], bins=20)
plt.xlabel('Flight Price')
plt.ylabel('Frequency')
plt.title('Distribution of Flight Prices')
plt.show()


# What is the range of prices in the dataset? What is the minimum and maximum price?

# In[15]:


import pandas as pd


df = pd.read_csv('flight_price.xlsx - Sheet1.csv')


price_range = df['Price'].max() - df['Price'].min()
min_price = df['Price'].min()
max_price = df['Price'].max()


print("Price range:", price_range)
print("Minimum price:", min_price)
print("Maximum price:", max_price)


# How does the price of flights vary by airline? Create a boxplot to compare the prices of different
# airlines.

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('flight_price.xlsx - Sheet1.csv')


plt.figure(figsize=(10,6))
plt.boxplot([df[df['Airline']=='Airline A']['Price'], 
             df[df['Airline']=='Airline B']['Price'], 
             df[df['Airline']=='Airline C']['Price']])
plt.xticks([1, 2, 3], ['Airline A', 'Airline B', 'Airline C'])
plt.xlabel('Airline')
plt.ylabel('Price')
plt.title('Flight Prices by Airline')
plt.show()


# Are there any outliers in the dataset? Identify any potential outliers using a boxplot and describe how
# they may impact your analysis.

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('flight_price.xlsx - Sheet1.csv')


plt.boxplot(df['Price'])
plt.ylabel('Price')
plt.title('Boxplot of Flight Prices')
plt.show()


# You are working for a travel agency, and your boss has asked you to analyze the Flight Price dataset
# to identify the peak travel season. What features would you analyze to identify the peak season, and how
# would you present your findings to your boss?

# To identify the peak travel season from the Flight Price dataset, we can analyze several features. Here are some of the features that can be analyzed:
# 
# Time of Year: Analyze the prices by month or quarter of the year to identify patterns in the data. This will help identify whether prices are higher during certain months or seasons.
# 
# Destination: Analyze the prices by destination to identify whether certain locations have higher prices during specific times of the year.
# 
# Day of the Week: Analyze the prices by day of the week to determine whether prices are higher on weekends or weekdays.
# 
# Flight Duration: Analyze the prices by flight duration to identify whether prices are higher during peak travel periods such as holidays or school breaks.
# 
# To present our findings to the boss, we can use different types of visualizations such as line plots, bar charts, or heatmaps. We can show the trends in flight prices over time and compare the prices for different seasons, destinations, days of the week, and flight durations. We can also use statistical methods such as regression analysis to identify any significant correlations between these variables and flight prices.

# You are a data analyst for a flight booking website, and you have been asked to analyze the Flight
# Price dataset to identify any trends in flight prices. What features would you analyze to identify these
# trends, and what visualizations would you use to present your findings to your team?

# To identify trends in flight prices, we can analyze several features. Here are some of the features that we can consider analyzing:
# 
# Time of Year: Analyze the prices by month or quarter of the year to identify any seasonal patterns in the data.
# 
# Day of the Week: Analyze the prices by day of the week to determine whether prices are higher on weekends or weekdays.
# 
# Flight Duration: Analyze the prices by flight duration to identify whether prices are higher for longer or shorter flights.
# 
# Departure and Arrival Cities: Analyze the prices by departure and arrival cities to identify whether prices vary by location.
# 
# Airline: Analyze the prices by airline to determine whether certain airlines have higher or lower prices than others.
# 
# To present our findings to the team, we can use different types of visualizations such as line plots, bar charts, or heatmaps. Here are some examples of how we can use these visualizations to present our findings:
# 
# Line Plot: We can create a line plot that shows the average flight prices by month for the entire dataset or for specific airlines or departure cities. This will help us identify any seasonal patterns in the data and see how prices fluctuate throughout the year.
# 
# Bar Chart: We can create a bar chart that shows the average flight prices by day of the week for the entire dataset or for specific airlines or departure cities. This will help us identify whether prices are higher on weekends or weekdays.
# 
# Heatmap: We can create a heatmap that shows the flight prices by departure and arrival cities. This will help us identify any patterns in the data based on location.
# 
# Scatter Plot: We can create a scatter plot that shows the relationship between flight duration and price. This will help us identify whether there is a correlation between these two variables.

# You are a data scientist working for an airline company, and you have been asked to analyze the
# Flight Price dataset to identify the factors that affect flight prices. What features would you analyze to
# identify these factors, and how would you present your findings to the management team?

# To identify the factors that affect flight prices, we can analyze several features in the Flight Price dataset. Here are some of the features that we can consider analyzing:
# 
# Time of Year: Analyze the prices by month or quarter of the year to identify whether prices are affected by seasonal patterns.
# 
# Day of the Week: Analyze the prices by day of the week to determine whether prices are affected by weekends or weekdays.
# 
# Flight Duration: Analyze the prices by flight duration to determine whether prices are affected by the length of the flight.
# 
# Departure and Arrival Cities: Analyze the prices by departure and arrival cities to identify whether prices vary by location.
# 
# Airline: Analyze the prices by airline to determine whether prices vary by airline.
# 
# Number of Stops: Analyze the prices by the number of stops to determine whether prices vary by the number of stops.
# 
# Time of Booking: Analyze the prices by the time of booking to determine whether prices are affected by how far in advance the flight is booked.
# 
# To present our findings to the management team, we can use a combination of statistical analysis and data visualization. Here are some examples of how we can present our findings:
# 
# Regression Analysis: We can perform regression analysis on the Flight Price dataset to identify which variables have a significant impact on flight prices. This will help us quantify the effect of each variable on the flight prices and determine which variables are the most important.
# 
# Correlation Matrix: We can create a correlation matrix that shows the correlation between each variable and flight prices. This will help us identify which variables are strongly correlated with flight prices and which variables are not.
# 
# Boxplots: We can create boxplots that show the distribution of flight prices by different variables such as airline, departure city, and number of stops. This will help us identify any differences in flight prices based on these variables.
# 
# Heatmaps: We can create heatmaps that show the flight prices by month and departure city. This will help us identify any seasonal patterns in the data and how flight prices vary by location.

# Load the Google Playstore dataset and examine its dimensions. How many rows and columns does
# the dataset have?

# In[21]:


import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')


print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# How does the rating of apps vary by category? Create a boxplot to compare the ratings of different
# app categories.

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')


df = df[['Category', 'Rating']]


grouped_df = df.groupby(['Category']).mean().reset_index()


plt.figure(figsize=(12,8))
plt.boxplot(grouped_df['Rating'])
plt.xticks([1], ['App Categories'])
plt.ylabel('Ratings')
plt.title('Boxplot of App Ratings by Category')
plt.show()


# Are there any missing values in the dataset? Identify any missing values and describe how they may
# impact your analysis.

# In[23]:


import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')


print(df.isnull().sum())


# What is the relationship between the size of an app and its rating? Create a scatter plot to visualize
# the relationship.

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')

df = df[['Size', 'Rating']]


df.dropna(inplace=True)




plt.figure(figsize=(12,8))
plt.scatter(df['Size'], df['Rating'])
plt.xlabel('App Size (in MB)')
plt.ylabel('App Rating')
plt.title('Relationship between App Size and Rating')
plt.show()


# How does the type of app affect its price? Create a bar chart to compare average prices by app type.

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')


df.dropna(inplace=True)


df = df[['Type', 'Price']]


df['Price'] = pd.to_numeric(df['Price'].apply(lambda x: x.replace('$', '')))


avg_price_by_type = df.groupby('Type')['Price'].mean()


plt.figure(figsize=(12,8))
plt.bar(avg_price_by_type.index, avg_price_by_type.values)
plt.xlabel('App Type')
plt.ylabel('Average Price ($)')
plt.title('Average Price by App Type')
plt.show()


# What are the top 10 most popular apps in the dataset? Create a frequency table to identify the apps
# with the highest number of installs.

# In[34]:


import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')


df.dropna(inplace=True)

df['Installs'] = df['Installs'].apply(lambda x: int(x.replace('+', '').replace(',', '')))

grouped_df = df.groupby('App')['Installs'].sum().reset_index()

sorted_df = grouped_df.sort_values('Installs', ascending=False)

top_10_apps = sorted_df.head(10)

freq_table = pd.Series(top_10_apps.Installs.values, index=top_10_apps.App.values)

print(freq_table)


# A company wants to launch a new app on the Google Playstore and has asked you to analyze the
# Google Playstore dataset to identify the most popular app categories. How would you approach this
# task, and what features would you analyze to make recommendations to the company?

# To identify the most popular app categories on the Google Playstore dataset, we can use a combination of data exploration, visualization, and statistical analysis. Here are the steps I would take to approach this task:
# 
# Load and clean the dataset: Load the Google Playstore dataset into a data analysis tool like Python or R, and clean the data by removing any missing or erroneous values.
# 
# Explore the dataset: Explore the dataset to understand its structure and identify the key features that can be used to analyze app popularity. Some of the features that could be used include Category, Rating, Reviews, Installs, and Price.
# 
# Visualize the data: Use visualization tools like histograms, bar charts, and scatter plots to visualize the distribution of each feature and identify any patterns or trends.
# 
# Analyze the data: Use statistical analysis techniques like correlation analysis and regression analysis to identify the relationship between the different features and app popularity. For example, we could analyze the correlation between app category and the number of installs or ratings.
# 
# Make recommendations: Based on the analysis, make recommendations to the company on the most popular app categories and any other insights that could help them make informed decisions on their new app launch.
# 
# Some of the specific features that I would analyze to make recommendations to the company include:
# 
# Category: Identify the most popular app categories based on the number of installs and ratings. This will help the company understand which categories are currently in high demand and could be a good target for their new app.
# 
# Reviews: Analyze the relationship between the number of reviews and app popularity. This will help the company understand how important user feedback is in driving app popularity and whether they need to focus on improving their app's review system.
# 
# Installs: Analyze the distribution of app installs across different categories and identify any trends or patterns. This will help the company understand the level of competition in each category and whether they need to focus on marketing their app more aggressively.
# 
# Price: Analyze the relationship between app price and app popularity. This will help the company understand whether users are willing to pay for premium apps in certain categories and whether they should consider a freemium or premium pricing model for their new app.

# A mobile app development company wants to analyze the Google Playstore dataset to identify the
# most successful app developers. What features would you analyze to make recommendations to the
# company, and what data visualizations would you use to present your findings?

# To identify the most successful app developers on the Google Playstore dataset, we can use a combination of data exploration, visualization, and statistical analysis. Here are the features I would analyze and the data visualizations I would use to present my findings:
# 
# Developer name: The first feature I would analyze is the developer name. I would group the data by developer and calculate the sum of installs, ratings, and reviews for each developer.
# 
# App category: The second feature I would analyze is the app category. I would group the data by developer and category and calculate the sum of installs, ratings, and reviews for each developer and category combination.
# 
# App price: The third feature I would analyze is the app price. I would group the data by developer and calculate the average price and the number of paid apps for each developer.
# 
# To present my findings, I would use the following data visualizations:
# 
# Bar charts: I would use bar charts to visualize the total number of installs, ratings, and reviews for each developer. This would allow us to quickly identify the most successful developers based on these metrics.
# 
# Heat maps: I would use heat maps to visualize the sum of installs, ratings, and reviews for each developer and category combination. This would allow us to identify the categories where each developer is most successful and whether there are any categories that are particularly strong for a given developer.
# 
# Scatter plots: I would use scatter plots to visualize the relationship between the average price of apps and the number of paid apps for each developer. This would allow us to identify whether there is a correlation between these two variables and whether developers who charge more for their apps tend to have more success.

# A marketing research firm wants to analyze the Google Playstore dataset to identify the best time to
# launch a new app. What features would you analyze to make recommendations to the company, and
# what data visualizations would you use to present your findings?

# To identify the best time to launch a new app on the Google Playstore dataset, we can use a combination of data exploration, visualization, and statistical analysis. Here are the features I would analyze and the data visualizations I would use to present my findings:
# 
# Release date: The first feature I would analyze is the release date of apps. I would group the data by release date and calculate the average number of installs and ratings for each release date.
# 
# App category: The second feature I would analyze is the app category. I would group the data by category and release date and calculate the average number of installs and ratings for each category and release date combination.
# 
# App size: The third feature I would analyze is the app size. I would group the data by app size and release date and calculate the average number of installs and ratings for each app size and release date combination.
# 
# To present my findings, I would use the following data visualizations:
# 
# Line charts: I would use line charts to visualize the trend in the average number of installs and ratings over time. This would allow us to identify any seasonal trends or patterns in app popularity that could influence the best time to launch a new app.
# 
# Heat maps: I would use heat maps to visualize the average number of installs and ratings for each category and release date combination. This would allow us to identify which categories tend to be more popular at certain times of the year and whether there are any particular release dates that are particularly strong for a given category.
# 
# Scatter plots: I would use scatter plots to visualize the relationship between app size and the average number of installs and ratings for each release date. This would allow us to identify whether there is a correlation between app size and app popularity and whether there are any particular release dates that are more favorable for larger or smaller apps.

# In[ ]:





# In[ ]:





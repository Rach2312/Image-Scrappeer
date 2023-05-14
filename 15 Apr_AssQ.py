#!/usr/bin/env python
# coding: utf-8

# You are work#ng on a mach#ne learn#ng project where you have a dataset conta#n#ng numer#cal and
# categor#cal features. You have #dent#f#ed that some of the features are h#ghly correlated and there are
# m#ss#ng values #n some of the columns. You want to bu#ld a p#pel#ne that automates the feature
# eng#neer#ng process and handles the m#ss#ng valuesD
# Des#gn a p#pel#ne that #ncludes the follow#ng steps"
# Use an automated feature select#on method to #dent#fy the #mportant features #n the datasetC
# Create a numer#cal p#pel#ne that #ncludes the follow#ng steps"
# Impute the m#ss#ng values #n the numer#cal columns us#ng the mean of the column valuesC
# Scale the numer#cal columns us#ng standard#sat#onC
# Create a categor#cal p#pel#ne that #ncludes the follow#ng steps"
# Impute the m#ss#ng values #n the categor#cal columns us#ng the most frequent value of the columnC
# One-hot encode the categor#cal columnsC
# Comb#ne the numer#cal and categor#cal p#pel#nes us#ng a ColumnTransformerC
# Use a Random Forest Class#f#er to bu#ld the f#nal modelC
# Evaluate the accuracy of the model on the test datasetD
# Note! Your solut#on should #nclude code sn#ppets for each step of the p#pel#ne, and a br#ef explanat#on of
# each step. You should also prov#de an #nterpretat#on of the results and suggest poss#ble #mprovements for
# the p#pel#neD

# In[37]:



from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

X = data.drop('target', axis=1) 
y = data['target']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numerical_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

categorical_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder()
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

model = RandomForestClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(f_classif, k=10)),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy}")


# Bu#ld a p#pel#ne that #ncludes a random forest class#f#er and a log#st#c regress#on class#f#er, and then
# use a vot#ng class#f#er to comb#ne the#r pred#ct#ons. Tra#n the p#pel#ne on the #r#s dataset and evaluate #ts
# accuracy.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100)
lr_clf = LogisticRegression()
pipeline = Pipeline([
    ('rf', rf_clf),
    ('lr', lr_clf)
])

voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('lr', lr_clf)],
    voting='soft')

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print("Pipeline accuracy: {:.2f}%".format(accuracy * 100))

voting_clf.fit(X_train, y_train)

accuracy = voting_clf.score(X_test, y_test)
print("Voting classifier accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:






# coding: utf-8

# In[1]:

import sys
import numpy as np
import pandas as pd
from skmultilearn.problem_transform import LabelPowerset
# from skmultilearn.problem_transform import BinaryRelevance
# from skmultilearn.problem_transform import ClassifierChain
# from skmultilearn.adapt.mlknn import MLkNN
# from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:

weather = pd.read_csv('weather_pd.csv')


# In[3]:

description = ['Freezing', 'Heavy', 'Mainly', 'Moderate', 'Mostly', 'Clear', 'Cloudy', 'Drizzle', 'Fog', 'Rain', 'Snow', 'Showers']
y = pd.DataFrame()
for label in description:
    y[label] = weather['Weather'].str.contains(label).astype(int)
X = weather.iloc[:, 5:12]
# X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[4]:

X_test = weather[weather['Year']>2016]
X_test = X_test[X_test['Month']>8].iloc[:, 5:12]
X_train = weather[~weather.index.isin(X_test.index)].iloc[:, 5:12]
y_test = y[y.index.isin(X_test.index)]
y_train = y[~y.index.isin(X_test.index)]


# In[5]:

# model = LabelPowerset(GaussianNB())
model = LabelPowerset(KNeighborsClassifier(n_neighbors=20))
# model = MLkNN(k=18)
model.fit(X_train.values, y_train.values)
predictions = model.predict(X_test)
result = predictions.toarray()
predicted = pd.DataFrame(result, columns = description)
print(accuracy_score(y_test,predictions))


# In[6]:

def columnName(row):
    name = ''
    idx = row[row==1].index
    for i in range (len(idx)):
        name += (idx[i] + ' ')
    return name


# In[7]:

predict_weather = predicted.apply(columnName, axis=1)
weather = weather[weather.index.isin(X_test.index)]
weather =weather.assign(predicted_weather = predict_weather.values)


# In[8]:

weather.to_csv('weather_predicton_by_conditions.csv', index=False)


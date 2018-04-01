
# coding: utf-8

# In[23]:

# Refering to k-NN classifier for image classification by Adrian Rosebrock
# https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
import cv2
import glob
import numpy as np
import pandas as pd
from skmultilearn.problem_transform import LabelPowerset
# from skmultilearn.problem_transform import BinaryRelevance
# from skmultilearn.problem_transform import ClassifierChain
# from skmultilearn.adapt.mlknn import MLkNN
# from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# In[30]:

weather = pd.read_csv('weather_pd.csv')
weather['Date/Time'] = pd.to_datetime(weather['Date/Time'])


# In[3]:

# method inspired by A.Feliz
# https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
imgs = []
dates = []
for img in glob.glob("katkam-scaled\katkam*.jpg"):
    image = cv2.imread(img)
    image = cv2.resize(image, dsize=(64,48)).flatten()
    time = pd.to_datetime(img[21:25]+'-'+img[25:27]+'-'+img[27:29]+' '+img[29:31])
    imgs.append(image)
    dates.append(time)


# In[4]:

katkam = pd.DataFrame(np.array(imgs))
katkam['Date/Time'] = dates
k_weather = pd.merge(weather, katkam, on='Date/Time')


# In[6]:

description = ['Freezing', 'Heavy', 'Mainly', 'Moderate', 'Mostly', 'Clear', 'Cloudy', 'Drizzle', 'Fog', 'Rain', 'Snow', 'Showers']
y = pd.DataFrame()
for label in description:
    y[label] = k_weather['Weather'].str.contains(label).astype(int)
X = k_weather.iloc[:, 13:]
# X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[7]:

X_test = k_weather[k_weather['Year']>2016]
X_test = X_test[X_test['Month']>8].iloc[:, 13:]
X_train = k_weather[~k_weather.index.isin(X_test.index)].iloc[:, 13:]
y_test = y[y.index.isin(X_test.index)]
y_train = y[~y.index.isin(X_test.index)]


# In[8]:

model = make_pipeline(
    PCA(300),
#     LabelPowerset(SVC(C=5.0))
    LabelPowerset(KNeighborsClassifier(n_neighbors=20))
#     MLkNN(k=20)
)
model.fit(X_train.values, y_train.values)
predictions = model.predict(X_test)
result = predictions.toarray()
predicted = pd.DataFrame(result, columns = description)
print(accuracy_score(y_test,predictions))


# In[19]:

def columnName(row):
    name = ''
    idx = row[row==1].index
    for i in range (len(idx)):
        name += (idx[i] + ' ')
    return name


# In[61]:

predict_weather = predicted.apply(columnName, axis=1)
weather = k_weather.iloc[:,0:13]
weather = weather[weather.index.isin(X_test.index)]
weather = weather.assign(predicted_weather = predict_weather.values)


# In[63]:

weather.to_csv('weather_prediction_by_images.csv', index=False)


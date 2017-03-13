
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Categorical Data

data = [{'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data) 






#Text

#Approach 1- CountVectorizer

sample = ['problem of evil','evil queen','horizon problem']

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)

pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


#Approach 2- TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())





#Trial (different transforms)

x = np.array([3,5,2,6,4,2,6,3,6])
X_trial = x[:,np.newaxis]
Y_trial = np.random.rand(9,)

X_test = np.array([3,5,2,0,19])

from sklearn.linear_model import LinearRegression
model = LinearRegression()
Y_model = model.fit(X_trial,Y_trial).predict(X_test)
plt.scatter(X_trial,Y_trial)
plt.plot(X_test,Y_model)


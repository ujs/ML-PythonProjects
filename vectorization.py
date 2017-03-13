
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# #Categorical Data

data = [{'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data) 






# #Text

# #Approach 1- CountVectorizer

sample = ['problem of evil','evil queen','horizon problem']

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)

pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


# #Approach 2- TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())





# #Trial (different transforms)

x = np.array([3,5,2,6,4,2])

Y_trial = np.array([5,2,6,8,3,5])



from sklearn.linear_model import LinearRegression
X_trial = x[:,np.newaxis]
X_test = X_trial
model = LinearRegression()
Y_model = model.fit(X_trial,Y_trial).predict(X_test)
# plt.scatter(X_trial,Y_trial)
# plt.plot(X_test,Y_model)


from sklearn.preprocessing import PolynomialFeatures
tran = PolynomialFeatures(degree=3, include_bias=False)
X_new = tran.fit_transform(X_trial)

Y_model = model.fit(X_new,Y_trial).predict(X_new)
# plt.scatter(x,Y_trial)
# plt.plot(x,Y_model)


# Shortcut using pipeline
from numpy import nan
from sklearn.preprocessing import Imputer
X = np.array([[ nan, 0,   3  ],
              [ 3,   7,   9  ],
              [ 3,   5,   2  ],
              [ 4,   nan, 6  ],
              [ 8,   8,   1  ]])
y = np.array([14, 16, -1,  8, -5])
from sklearn.pipeline import make_pipeline

model = make_pipeline(Imputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())

model.fit(X, y)
print(model.predict(X))


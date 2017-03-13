import numpy as np 
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


def make_data(N, err = 1.0, rseed = 1):
	#random sampling
	rng = np.random.RandomState(rseed)
	X = rng.rand(N,1)**2
	y = 10 - 1. / (X.ravel() + 0.1)
	if err > 0:
		y += err * rng.randn(N)
	return X, y

X, y = make_data(50)

X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')

for degree in [1,2,4,6]:
	y_test = PolynomialRegression(degree).fit(X,y).predict(X_test)
# 	plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
# plt.xlim(-0.1, 1.0)
# plt.ylim(-2, 12)
# plt.legend(loc='best')

# Comparing validation scores and training scores across degree

from sklearn.learning_curve import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          'polynomialfeatures__degree', degree, cv=7)

plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')

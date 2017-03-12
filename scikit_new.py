import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
# sns.pairplot(iris, hue = 'species',kind ='scatter')

Y_iris = iris['species']
X_iris = iris.drop('species', axis = 1)

# ML steps
# Step One- Import class that corresponds to modeling method

from sklearn.linear_model import LinearRegression

#Step 2 - Instantiation & choosing parameters

model = LinearRegression(fit_intercept = True)

#Step 3 - Arrange Data into Features & Target Matrix

rng = np.random.RandomState(42)
x = rng.rand(50)
y = 2*x + 1 + rng.rand(50)
X = x[:,np.newaxis]

# Step 4- Fit model to data
model.fit(X,y)
c = model.coef_
i = model.intercept_

# print('Model Coefficient: ' + str(c))
# print('Model intercept: ' + str(i))

# Step 5. predicting unknown values of y based on the training
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter (x,y)
plt.plot(xfit, yfit)
plt.axis([0,5,0,5])




# Applications of ML on Iris Data


# 1. Naive Bayes Classification- Supervised Learning
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, Y_iris, random_state = 1)

from sklearn.naive_bayes import GaussianNB # Import class
model = GaussianNB() 					   # Instantiate
model.fit(Xtrain, ytrain)				   # Fit model to data
y_model = model.predict(Xtest)			   # Predict

#Check Accuracy of Model's prediction


from sklearn.metrics import accuracy_score  #this has deprecated
# print(accuracy_score(ytest, y_model))	


# 2. PCA- Dimensionality (unsupervised)
from sklearn.decomposition import PCA
model = PCA(n_components = 2)
model.fit(X_iris)
X_2D = model.transform(X_iris)

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)

# 3. GMM- Unsupervised
from sklearn.mixture import GMM
model = GMM (n_components = 3, covariance_type = 'full' )
model.fit(X_iris)                   
y_gmm = model.predict(X_iris)  
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species', col='cluster', fit_reg=False)






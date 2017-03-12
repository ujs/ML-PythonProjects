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

# plt.scatter (x,y)
# plt.plot(xfit, yfit)
# plt.axis([0,5,0,5])




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
# sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)


# 3. GMM- Unsupervised    
from sklearn.mixture import GaussianMixture
model = GaussianMixture (n_components = 3, covariance_type = 'full' )
model.fit(X_iris)                   
y_gmm = model.predict(X_iris)  
iris['cluster'] = y_gmm
# sns.lmplot("PCA1", "PCA2", data=iris, hue='species', col='cluster', fit_reg=False)



# Applications of ML on Digits Data
from sklearn.datasets import load_digits
digits = load_digits()

fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))


for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')

# Dimensionality (unsupervised)

from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape

plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)

# Naive Bayes (Supervised)

X = digits.data
y = digits.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

from sklearn.naive_bayes impot GaussianNB
model = GaussianNB()
model.fit(Xtrain,ytrain)
y_model = model.predict(Xtest)

# Confusion matrix to see where the Gaussian model messed up
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('actual value')



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







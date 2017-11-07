import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd 

profit_data = pd.read_csv ('/Users/ujalashanker/Downloads/machine-learning-ex1/ex1/ex1data1.txt', header = None, names=['Population','Profit'])
vals = profit_data.values # Converts data frame to numpy array
profit_data.describe()
plt.scatter(vals[:, 0], vals[:, 1])		#plotting the data 
plt.title('Scatterplot of Training Data')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
# plt.show()




# iterations = 1500		#setting the number of gradient descent iterations
# alpha = 0.01			#setting the alpha parameter
                           
def ComputeCost(X,y,theta):
	inner = np.power(((X*theta.T)-y),2)
	return np.sum(inner)/(2*len(X))

profit_data.insert(0, 'Ones',1) #Inserting the coefficients for theta-0

cols = profit_data.shape[1]
X = profit_data.iloc[:,0:cols-1]
y = profit_data.iloc[:,cols-1:cols]

#converting from data frame to matrices
# X= np.matrix(X.values)
# y= np.matrix(y.values)
# theta = np.matrix(np.array([0,0]))
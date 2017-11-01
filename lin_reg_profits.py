import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd 

profit_data = pd.read_csv ('/Users/ujalashanker/Downloads/machine-learning-ex1/ex1/ex1data1.txt', header = None, names=['Population','Profit'])
vals = profit_data.values
profit_data.describe()
plt.scatter(vals[:, 0], vals[:, 1])		#plotting the data 
plt.title('Scatterplot of Training Data')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


theta = np.zeros(shape = (2,1)) #initializing the parameter values
iterations = 1500		#setting the number of gradient descent iterations
alpha = 0.01			#setting the alpha parameter
                           

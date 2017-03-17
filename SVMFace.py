import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)

#Examining few images
fig, ax = plt.subplots(3,5)
for i, axi in enumerate(ax.flat):
	axi.imshow(faces.images[i],cmap = 'bone')
	axi.set(xticks = [], yticks = [], xlabel = faces.target_names[faces.target[i]])

#Each Image has 3K pixels. So will use PCA to extract main 150 features
#Importing libraries (Classes) for SVM, PCA and pipelines

from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline

#Instantiating the imported classes

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

#preparing pur training and test data

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

#Comparing the performance of the model for different parameters(C- margin hardness, gamma- size of the radial basis function kernel)
from sklearn.grid_search import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

grid.fit(Xtrain, ytrain)
print(grid.best_params_)
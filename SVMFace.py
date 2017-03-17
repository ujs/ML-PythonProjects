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
#Importing libraries (Classes)

from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

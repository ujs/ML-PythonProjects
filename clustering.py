# Example 1- Identifying Digits
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# Load Data
from sklearn.datasets import load_digits


# Clustering

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0)
X = load_digits().data
# clusters = kmeans.fit_predict(X)
# # print(kmeans.cluster_centers_)

# #Assigning Labels

# # from scipy.stats import mode
# # labels = np.zeros_like(clusters)


# # Example 2- k-means for color compression
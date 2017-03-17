# Example 1- Identifying Digits
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# Load Data
from sklearn.datasets import load_digits
digits = load_digits

#Clustering

from sklearn.cluster import KMeans
clusters = kmeans.fit_predict(digits.data)
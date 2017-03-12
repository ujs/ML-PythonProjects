import pandas as pd 
import numpy as np
import seaborn as sns

iris = sns.load_dataset('iris')
# sns.pairplot(iris, hue = 'species',kind ='scatter')

target_vector = iris['species']
features_matrix = iris.drop(iris.species)



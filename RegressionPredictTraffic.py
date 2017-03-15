import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

counts = pd.read_csv('FremontBridge.csv', index_col = 'Date', parse_dates = True)
weather =pd.read_csv('weatherSeattle.csv', index_col = 'DATE', parse_dates = True)

data = counts.resample('d', how='sum')
data['Total'] = data.sum(axis =1)
data = data['Total']
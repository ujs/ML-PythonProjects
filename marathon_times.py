import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

run_data = pd.read_csv('marathon-data.csv')
run_data['split'] = pd.to_timedelta(run_data['split'])
run_data['final'] = pd.to_timedelta(run_data['final'])

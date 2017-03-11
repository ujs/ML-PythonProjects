import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

run_data = pd.read_csv('marathon-data.csv')
run_data['split'] = pd.to_timedelta(run_data['split'])
run_data['final'] = pd.to_timedelta(run_data['final'])

run_data['split_sec'] = run_data['split'].astype(int)/1E9
run_data['final_sec'] = run_data['final'].astype(int)/1E9
 # Split strategy calculates if runner had a negative split or positive split during the race
 # A negative value of 'split_strategy' means negative split and positive value means positive split
run_data['split_strategy'] = run_data['final_sec'] - 2* run_data['split_sec']

sns.distplot(run_data['split_strategy'], kde = 'False')
plt.axvline(0, color = 'k', linestyle = '--')
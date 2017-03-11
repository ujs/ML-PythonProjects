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
print (sum(run_data.split_strategy < 0))

grid_data = sns.PairGrid(run_data, vars = ['age', 'split_sec', 'final_sec', 'split_strategy'], hue = 'gender', palette='RdBu_r')
grid_data.map(plt.scatter, alpha = 0.9)
grid_data.add_legend()

# Male vs Female split strategy
#Observation - there are many more men than women who are running close to an even split
sns.kdeplot(run_data.split_strategy[run_data.gender == 'M'], label='men', shade = 'True')
sns.kdeplot(run_data.split_strategy[run_data.gender == 'F'], label='women', shade = 'True')
plt.xlabel('split_strategy')

#Violin Plot
sns.violinplot("gender", "split_strategy", data=run_data, palette=["lightblue", "lightpink"])

#Analysis By age

run_data['age_dec'] = run_data.age.map(lambda age: 10 * (age // 10))

men = (run_data.gender== 'M') 
women = (run_data.gender== 'F') 

# sns.violinplot("age_dec", "split_strategy", hue="gender", data=run_data, split=True, inner="quartile", palette=["lightblue", "lightpink"])


g = sns.lmplot('final_sec', 'split_strategy', col='gender', data=run_data,
               markers=".", scatter_kws=dict(color='c'))
g.map(plt.axhline, y=0.1, color="k", ls=":")


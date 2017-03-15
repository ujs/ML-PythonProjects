import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

counts = pd.read_csv('FremontBridge.csv', index_col = 'Date', parse_dates = True)
weather =pd.read_csv('weatherSeattle.csv', index_col = 'DATE', parse_dates = True)

data = counts.resample('d', how='sum')
data['Total'] = data.sum(axis =1)
data = data['Total']

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
	data[days[i]] = (data.index.day == i).astype(float)

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
holidays = pd.Series(1, index = holidays, name='holiday')
hl.reshape()
data = data.join(holidays)
data['holiday'].fillna(0, inplace=True)
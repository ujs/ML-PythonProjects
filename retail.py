import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
retail_data = pd.read_csv('SampleRetail.csv')
Y = retail_data['OnTime/Not']
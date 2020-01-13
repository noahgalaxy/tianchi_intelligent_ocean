import numpy as np
import pandas as pd
from PIL import Image
import lightgbm as lgb
import os, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

'''
a = {'A':[1, 2, 3, 4], 'B':[5, 6, 7, 8]}
df = pd.DataFrame(a)
print(df)
li_a = np.array([9, 9, 9, 9])
print(li_a.shape)
#li_a = [9, 9, 9, 9]
df['C'] = li_a
print(df)

'''
li_a = [4, 9, 9, 7]
li_b = [7, 8, 8, 0]
ser_b = pd.Series(li_b)
ser_a = pd.Series(li_a)
print(ser_b)
b = pd.DataFrame(li_a, li_b, columns= ['id', 'pred'])
print(b)
'''
b = pd.concat([ser_a, ser_b], axis= 1)
b.columns = ['id', 'pre']
'''
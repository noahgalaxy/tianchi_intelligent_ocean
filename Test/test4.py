import numpy as np
import pandas as pd
from PIL import Image
import lightgbm as lgb
import os, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', None)


root_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102'
suffix = r'.csv'
seq = range(15)
csvs_path = [root_path + '\\' + str(i) +suffix for i in seq]
df = []
for path in csvs_path:
    df_temp = pd.read_csv(path)
    df.append(df_temp)

df = pd.concat(df, axis= 0)
df.columns = ['ship', 'x', 'y', 'v', 'd', 'time', 'type']
print(df.shape)
print(df.columns)
# create groupby object
df1 = df.groupby(by = 'ship')
#print(df.groups)
#print(df1['x'].agg(np.size))
direction = lambda x: 180. - x
def direct(df):
    bins = range(0, 361, 30)
    return pd.cut(df['d'], bins= bins).value_counts()
print(df1.apply(direct))
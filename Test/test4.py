import numpy as np
import pandas as pd
from PIL import Image
import lightgbm as lgb
import os, warnings
from matplotlib import pyplot as plt
import seaborn as sns
import math

LABEL_DICT = {
    '围网': 0,
    '拖网': 1,
    '刺网': 2
}

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', None)


root_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102'
suffix = r'.csv'
seq = range(7000)
csvs_path = [root_path + '\\' + str(i) +suffix for i in seq]
df = []
for path in csvs_path:
    df_temp = pd.read_csv(path)
    df.append(df_temp)

df = pd.concat(df, axis= 0)
df.columns = ['ship', 'x', 'y', 'v', 'd', 'time', 'type']
df['type'].map(LABEL_DICT)
print(df.shape)
print(df.columns)
# create groupby object
df1 = df.groupby(by = 'ship')
#print(df.groups)
#print(df1['x'].agg(np.size))

def direct(df):

    bins = range(0, 361, 30)
    labels = ['d_' + str(i + 1) + '_level' for i, section in enumerate(bins) if i < 12]
    print(labels)
    return pd.cut(df['d'], bins= bins, labels= labels).value_counts()
#print(df1.apply(direct))

'''
print('围网', new_df[new_df['type'] == '围网']['sum'].mean())
print('拖网', new_df[new_df['type'] == '拖网']['sum'].mean())
print('刺网', new_df[new_df['type'] == '刺网']['sum'].mean())
'''
'''

weiwang = np.array(new_df[new_df['type'] == '围网']['sum'].values)
print(weiwang.shape)
tuowang = np.array(new_df[new_df['type'] == '拖网']['sum'].values)
ciwang = np.array(new_df[new_df['type'] == '刺网']['sum'].values)

print('weiwang shape', weiwang.shape)
print('tuowang shape', tuowang.shape)
print('ciwang shape', ciwang.shape)
x_weiwang = 1 * np.ones(len(weiwang))
x_tuowang = 2 * np.ones(len(tuowang))
x_ciwang = 3 * np.ones(len(ciwang))



plt.scatter(x_weiwang, weiwang, c= 'r', s = 5)
plt.scatter(x_tuowang, tuowang, marker= 'x', c= 'b', s = 5)
plt.scatter(x_ciwang, ciwang, marker= 'o', c= 'g', s = 5)
plt.show()
'''
plt.figure(figsize= (25, 15))
for i in range(7):
    def direction(df):
        df_1 = abs(np.sin(df['d']/ 360. + (math.pi / 6) * i).sum())
        return df_1
    def tupe_1(df):
        df_2 = df['type'].unique()
        return df_2[0]

    new_df = pd.concat([df1.apply(direction), df1.apply(tupe_1)], axis= 1)
    new_df.columns = ['sum', 'type']
    weiwang = np.array(new_df[new_df['type'] == '围网']['sum'].values)

    tuowang = np.array(new_df[new_df['type'] == '拖网']['sum'].values)
    ciwang = np.array(new_df[new_df['type'] == '刺网']['sum'].values)
    plt.subplot(2, 4, i+1)
    sns.stripplot(x='type', y='sum', data=new_df)
#print(new_df.head())
plt.show()


import numpy as np
from glob import glob
import pandas as pd
from PIL import Image
import lightgbm as lgb


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 20)

def make_hdf():
    root_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102\*csv'
    temp = []
    for i, csv in enumerate(glob(root_path)):
        print(i)
        temp.append(pd.read_csv(csv))
    df = pd.concat(temp, axis= 0)
    # 存储
    df.to_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\train.h5', 'df', mode='w')
    print('done')

def get_invalid_ship():
    train_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\train.h5'
    test_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\test.h5'
    DF = pd.read_hdf(train_path, 'df')
    #df = pd.read_hdf(test_path, 'df')
    DF.columns = ['ship', 'x', 'y', 'v', 'd', 'time', 'type']
    #df.columns = ['ship', 'x', 'y', 'v', 'd', 'time']
    print(DF.head())
    df = DF.groupby(by= 'ship')
    def d(df):
        return (df[['d', 'v']].sum())
    df1 = df.apply(d)
    print(df1)
    df_t = df1[(df1['d'] < 300) & (df1['v'] < 20)]
    print('无效数据数量:\n', df_t.count())
    # 得到无效数据ship号
    print('ship index', df_t.index)
    invalid_ship = df_t.index.tolist()
    print('invalid', invalid_ship)

if __name__ == '__main__':

    #make_hdf()
    get_invalid_ship()
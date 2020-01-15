
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import tables
def make_hdf():
    warnings.filterwarnings('ignore')
    train_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102'
    test_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_testA_20200102'

    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)
    print(len(train_files), len(test_files))

    ret = []
    for file in tqdm(train_files):
        df = pd.read_csv(f'{train_path}/{file}')
        ret.append(df)
    df = pd.concat(ret)
    df.columns = ['ship','x','y','v','d','time','type']
    print(df.info())
    df.to_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\train.h5', 'df', mode='w')
    print('train df', df.shape)

    ret = []
    for file in tqdm(test_files):
        df = pd.read_csv(f'{test_path}/{file}')
        ret.append(df)
    df = pd.concat(ret)
    df.columns = ['ship', 'x', 'y', 'v', 'd', 'time']
    df.to_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\test.h5', 'df', mode='w')
    print('test df', df.shape)

make_hdf()

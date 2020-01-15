import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


def group_feature(df, key, target, aggs):
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t


def extract_feature(df, train):
    '''
    此函数是给输入的df添加新的特征，分别有：
    (x,y,d,v)的(max,min,mean,std,skew,sum);
    slope、area、mode_hour、hour_max、hour_min、hour_nunique、date_nunique、diff_time、diff_day、diff_second
    加上原来的特征共49个特征
    '''
    t = group_feature(df, 'ship', 'x', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship', 'x', ['count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship', 'y', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship', 'v', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship', 'd', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['x_max_y_min'] = train['x_max'] - train['y_min']
    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min'] == 0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']

    mode_hour = df.groupby('ship')['hour'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_hour'] = train['ship'].map(mode_hour)

    t = group_feature(df, 'ship', 'hour', ['max', 'min'])
    train = pd.merge(train, t, on='ship', how='left')

    hour_nunique = df.groupby('ship')['hour'].nunique().to_dict()
    date_nunique = df.groupby('ship')['date'].nunique().to_dict()
    train['hour_nunique'] = train['ship'].map(hour_nunique)
    train['date_nunique'] = train['ship'].map(date_nunique)

    t = df.groupby('ship')['time'].agg({'diff_time': lambda x: np.max(x) - np.min(x)}).reset_index()
    t['diff_day'] = t['diff_time'].dt.days
    t['diff_second'] = t['diff_time'].dt.seconds
    train = pd.merge(train, t, on='ship', how='left')
    return train


def extract_dt(df):
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    # df['month'] = df['time'].dt.month
    # df['day'] = df['time'].dt.day
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    # df = df.drop_duplicates(['ship','month'])
    df['weekday'] = df['time'].dt.weekday
    return df

def make_train_test_label():
    train = pd.read_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\train.h5')
    test = pd.read_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\test.h5')
    test = extract_dt(test)
    train = extract_dt(train)
    # print(train.head())
    train_label = train.drop_duplicates('ship')
    print(train_label.shape)
    print(train_label.columns)
    test_label = test.drop_duplicates('ship')
    temp = train_label['type'].value_counts(normalize= True)
    print(temp)
    # 得到{'拖网': 0, '围网': 1, '刺网': 2}
    type_map = dict(zip(train_label['type'].unique(), np.arange(3)))
    print(type_map)
    # 反转标签k,v值，反转这个{'拖网': 0, '围网': 1, '刺网': 2}
    type_map_rev = {v:k for k,v in type_map.items()}
    # 将train_label['type']里面的type修改为0，1，2表示
    train_label['type'] = train_label['type'].map(type_map)

    train_label = extract_feature(train, train_label)
    test_label = extract_feature(test, test_label)
    print(train_label.columns)
    test_label.to_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\test_label.h5', key= 'df', mode= 'w')
    print('test save success!')
    train_label.to_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\train_label.h5', key= 'df', mode= 'w')
    print('train save success!')

def train_predict():
    train_label = pd.read_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\train_label.h5', key= 'df')
    test_label = pd.read_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\test_label.h5', key= 'df')
    features = [x for x in train_label.columns if x not in ['ship','type','time','diff_time','date']]
    target = 'type'
    print(len(features), ','.join(features))

    params = {
        'n_estimators': 5000,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'early_stopping_rounds': 100,
    }

    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X = train_label[features].copy()
    # 仅得到标签组成的df
    y = train_label[target]
    models = []
    pred = np.zeros((len(test_label),3))
    # 2000x3
    print('pred', pred.shape)
    oof = np.zeros((len(X), 3))
    # 7000x3
    print('oof', oof.shape)


    # 对每一折遍历
    for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):
        print(f'{index+1} start!!')
        # fold.split(X, y)返回的是分割后的索引的生成器
        # 制作lgb的训练集和验证集
        train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

        model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=100)
        models.append(model)
        val_pred = model.predict(X.iloc[val_idx])
        oof[val_idx] = val_pred
        val_y = y.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        print(index, 'val f1', metrics.f1_score(val_y, val_pred, average='macro'))
        # 0.8695539641133697
        # 0.8866211724839532

        test_pred = model.predict(test_label[features])
        pred += test_pred/5
    label_dict = {0: '拖网', 1:'围网', 2: '刺网'}
    pred = np.argmax(pred, axis=1)
    sub = test_label[['ship']]
    sub['pred'] = pred
    sub['pred'] = sub['pred'].map(label_dict)
    print(sub['pred'])
    print(sub['pred'].value_counts(1))
    print(sub.shape)
    print('=='*22)
    sub.to_csv(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\tianchi_ship_2019-master\dataset\result.csv', index= False, header= None)

if __name__ == '__main__':
    train_predict()
    #make_train_test_label()
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
from  CONFIG.CONFIG import FEATURES, FEATURES_1
import copy



def train_predict():
    features = FEATURES_1.copy()

    for temp in ['ship', 'type']:
        features.remove(temp)

    train_data = pd.read_hdf(
        r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\train_reset.h5', key='df')
    #train_label.drop('index', axis=1, inplace=True)

    #train_label.drop('ship', axis=1, inplace=True)
    test_data = pd.read_hdf(
        r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\test_reset.h5', key='df')
    #test_label.drop('index', axis=1, inplace=True)
    # 找出标签和船号
    train_label = train_data['type']
    test_ship = test_data['ship']
    # 去除train_data, test_data里面的'ship'和'type'
    train_data = train_data[features]
    test_data = test_data[features]

    '''
    params = {
    "learning_rate": 0.4,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "max_depth": 7,
    "num_leaves": 120,
    "objective": "multiclass",
    "num_class": 3,
    "verbose": -1,
    'feature_fraction': 0.8,
    "min_split_gain": 0.1,
    "boosting_type": "gbdt",
    "subsample": 0.8,
    "min_data_in_leaf": 50,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "tree_method": 'exact'
    }
    '''
    '''
    params = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'max_depth': 6,
        'num_leaves': 50
    }
    '''
    params = {
        'n_estimators': 5000,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'early_stopping_rounds': 100,
    }


    fold = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)
    models = []
    pred = 0

    print('pred', pred)


    # 对每一折遍历
    for index, (train_idx, val_idx) in enumerate(fold.split(train_data, train_label)):
        print(f'{index+1} start!!')
        # fold.split(X, y)返回的是分割后的索引的生成器
        # 制作lgb的训练集和验证集
        train_set = lgb.Dataset(train_data.iloc[train_idx], train_label.iloc[train_idx])
        val_set = lgb.Dataset(train_data.iloc[val_idx], train_label.iloc[val_idx])

        model = lgb.train(params, train_set, valid_sets= val_set, verbose_eval=100)
        models.append(model)
        val_pred = model.predict(train_data.iloc[val_idx])

        val_y = train_label.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        print(index, 'val f1', metrics.f1_score(val_y, val_pred, average='macro'))

        test_pred = model.predict(test_data)
        pred += test_pred/5

    label_dict = {0: '围网', 1:'拖网', 2: '刺网'}
    pred = np.argmax(pred, axis=1)
    # 将得到的np数组做成pd.Serise,便于后面拼接
    pred = pd.Series(pred)
    # 拼接Series
    test_pred = pd.concat([test_ship, pred], axis= 1)
    test_pred.columns = ['ship', 'pred']
    # 标签转换
    test_pred['pred'] = test_pred['pred'].map(label_dict)
    print(test_pred)
    print(test_pred['pred'].value_counts(normalize= True))
    print(test_pred.shape)
    print('=='*22)
    test_pred.to_csv(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\Results\result__2.csv', index= False, header= None)
if __name__ == '__main__':
    train_predict()
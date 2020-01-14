import numpy as np
import pandas as pd
from PIL import Image
import lightgbm as lgb
import os, warnings
from CONFIG.CONFIG import FEATURES, FEATURES_1
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', None)

test_hdf = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\test_reset.h5'
train_hdf = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\train_reset.h5'

features = FEATURES.copy()

for temp in ['ship', 'type']:
    features.remove(temp)

train_data = pd.read_hdf(train_hdf, key='df')

test_data = pd.read_hdf(test_hdf, key='df')
# test_label.drop('index', axis=1, inplace=True)
# 找出标签和船号
train_label = train_data['type']
print(train_label.values.shape)


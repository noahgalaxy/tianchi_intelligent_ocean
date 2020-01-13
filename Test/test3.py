import numpy as np
import pandas as pd
from PIL import Image
import lightgbm as lgb
import os, warnings
from CONFIG.CONFIG import FEATURES, FEATURES_1
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', None)

features = FEATURES_1.copy()
features.remove('type')
'''
for temp in ['ship', 'type']:
    features.remove(temp)
'''
train_label = pd.read_hdf(
    r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\train_reset.h5', key='df')
#train_label.drop('index', axis=1, inplace=True)
train_label = train_label[features]

#train_label.drop('ship', axis=1, inplace=True)
test_label = pd.read_hdf(
    r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\test_reset.h5', key='df')
#test_label.drop('index', axis=1, inplace=True)
#test_label = test_label[features]
#print(train_label.shape)
#print(test_label.index)
#print(test_label.head(10))
#print(test_label['ship'])
#print(test_label.ship)
a = 0
b = np.array([2, 5])
print(a + b)
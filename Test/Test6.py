import numpy as np
from glob import glob
import pandas as pd
from PIL import Image
import lightgbm as lgb
from CONFIG.CONFIG import FEATURES, FEATURES_1
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', None)
features = FEATURES.copy()

for temp in ['ship', 'type']:
    features.remove(temp)

train_data = pd.read_hdf(
    r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\train_modified_reset.h5', key='df')
# train_label.drop('index', axis=1, inplace=True)

# train_label.drop('ship', axis=1, inplace=True)
test_data = pd.read_hdf(
    r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\test_modified_reset.h5', key='df')
# test_label.drop('index', axis=1, inplace=True)
# 找出标签和船号
print(train_data.columns)

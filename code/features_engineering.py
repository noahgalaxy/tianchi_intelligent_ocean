import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import os, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

LABEL_DICT = {
    '围网': 0,
    '拖网': 1,
    '刺网': 2
}
COLUMNS = ['ship_id', '']
train_csvs_dir = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102'
test_csvs_dir = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_testA_20200102'
train_csvs_dir = train_csvs_dir + r'\*.csv'
test_csvs_dir = test_csvs_dir + r'\*.csv'

def get_8_values(df, keys, fun_list, only_process_v= False):
    sep = '_no_less' if only_process_v else ''
    total_dict = {}
    for key in keys:
        agg_dict = {}
        if key != 'ship':
            for fun in fun_list:
                agg_dict[f'{key}{sep}_{fun}'] = fun
            total_dict[key] = agg_dict

    #print(total_dict)
    temp = []
    df = df[keys]
    for key in keys:
        if key != 'ship':
            common_features = df.groupby(by= ['ship'])[key].agg(total_dict[key]).reset_index()
            temp.append(common_features)
    # 拼接
    common_features = temp[0]
    for i in range(len(temp))[1:]:
        common_features = pd.merge(common_features, temp[i], on= 'ship', how= 'left')
    sub_keys = keys[1:]
    for key in sub_keys:
        temp = common_features.loc[0, key + sep+'_max'] - common_features.loc[0,key + sep +'_min']
        common_features[key + sep + '_max_min'] = temp
        common_features[key + sep + '_max_min_mean'] = temp / common_features.loc[[0], [key + sep + '_mean']]
        common_features[key + sep + '_max_min_std'] = temp / common_features.loc[[0], [key + sep + '_std']]
    return common_features
def cut_df_into_10_(df):
    '''
    将x,y,v划分成10个档，找出每个档次所占的比例，每个档次的数量，
    将方向d划分成12个桶，返回每一档次所占的比例，每一档次的数量
    :return: 返回有这些新出来的特征组成的一个df，index为0， columns为特征名字，shape为 1xn
    '''
    features_list = ['x', 'y', 'v']
    total = []
    for feature in features_list:
        labels = []
        for i in range(10):
            labels.append(feature + '_' + str(i+1) + '_level')

        df1 = pd.cut(df[feature], bins= 10, labels= labels)

        #print(df1)
        df2 = df1.value_counts(normalize = True)
        df2.index = df2.index.map(lambda x: x + '_prop')
        df3 = df1.value_counts()
        df3.index = df3.index.map(lambda x: x + '_num')
        #print(df2)
        #print(df3)
        #df2 = pd.DataFrame(df2.values.T, index= ['prop'], columns= df2.index)
        #print('+'*22)
        #print(df2.index.values.tolist())
        df2 = pd.DataFrame((df2.values).reshape(1, 10), index= [0], columns= df2.index.values.tolist())
        df2 = df2[[x + '_prop' for x in labels]]
        #print(df2)
        df3 = pd.DataFrame((df3.values).reshape(1, 10), index= [0], columns= df3.index.values.tolist())
        df3 = df3[[x + '_num' for x in labels]]
        #print('+'*22)
        #print(df3)
        #print('=='*22)
        total.append(df2)
        total.append(df3)
        #total.append(pd.concat([df2, df3], axis= 1))

    # 处理方向d的切分，切分成bins_num档
    bins_num = 12
    labels = ['d_' + str(i+1) + '_level' for i in range(bins_num)]
    df_d = pd.cut(df['d'], bins= bins_num, labels= labels)
    for suffix in ['_num', '_prop']:
        df_d_suffix = df_d.value_counts(normalize= True) if suffix == '_prop' else df_d.value_counts()
        df_d_suffix.index = df_d_suffix.index.map(lambda x:x + suffix)
        # 将所得的Series的index换做columns，仅一条，所以index设为0
        df_d_suffix = pd.DataFrame((df_d_suffix.values).reshape(1, bins_num), index= [0], columns= df_d_suffix.index.values.tolist())
        df_d_suffix = df_d_suffix[[x + suffix for x in labels]]
        total.append(df_d_suffix)
    # 最后合并
    total = pd.concat(total, axis= 1)
    #print(total.shape)
    return total

def process_v(df):
    '''
    处理速度v的某些单独的：
    速度小于0.38的个数，比例；
    速度大于0.38的比例（1 - 上面的）；
    速度为0的个数，比例；
    速度不为0的比例（1 - 上面的）；
    然后再算出那些速度 v>0.38 的剩下的样本的那9个特征，用上面的 get_8_values 函数
    :return:
    返回由这些新构成的特征为columns的，index=0的df，shape为1xn；
    注意这里面去掉了ship这一项，即没有船的编号，主要是为了后面方便拼接的时候columns里面的值唯一
    '''
    columns_list = ['v_less_num', 'v_less_prop', 'v_no_less_prop', 'v_zero_num', 'v_zero_prop', 'v_no_zero_prop']
    df_num =df['v'].value_counts()
    df_prop = df['v'].value_counts(normalize= True)
    v_less_num, v_less_prop = (df_temp[df_temp.index.where(df_temp.index <= 0.38)].dropna().sum() for df_temp in [df_num, df_prop])
    v_no_less_prop = 1 - v_less_prop
    #print('速度v小于0.5的个数和比例, 和大于的比例：', v_less_num, v_less_prop, v_no_less_prop)
    #print('=='*22)
    try:
        v_zero_num, v_zero_prop = (df_num.loc[[0.00], ].values[0], df_prop.loc[[0.00], ].values[0])
    except:
        v_zero_num, v_zero_prop = 0.00001, 0.00001

    v_no_zero_prop = 1- v_zero_prop
    #print('速度为0的个数和比例，和大于的比例', v_zero_num, v_zero_prop, v_no_zero_prop)
    values_list = [v_less_num, v_less_prop, v_no_less_prop, v_zero_num, v_zero_prop, v_no_zero_prop]

    #print(df.shape)
    df_no_less = df['v'][df['v'] > 0.38]
    #print(df_no_less)
    #print('=='*22)
    #print(pd.DataFrame(dict(zip(columns_list, values_list)), index= [0]))
    #print('=='*22)

    fun_list = ['max', 'min', 'mean', 'std', 'skew', 'sum']
    keys = ['ship', 'v']
    v_no_less_8_common_feature = get_8_values(df, keys= keys, fun_list= fun_list, only_process_v= True)
    new_df = pd.concat([v_no_less_8_common_feature, pd.DataFrame(dict(zip(columns_list, values_list)), index= [0])], axis= 1)
    #print(new_df)
    #print(new_df.shape)
    # 去除这里面的ship列，免得后面合并的时候有多个ship列
    new_df.drop(['ship'], axis=1, inplace=True)
    return new_df

def extract_date_time(df):
    '''
    处理df里面的time这一项
    :return:
    返回数据持续的天数组成的df，index= 0, columns = 'lasting_days', shape = 1x1
    '''
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    # df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    #
    lasting_days = (df['day'].max() - df['day'].min() + 1)
    #df['lasting_days'] = lasting_days

    #df['date'] = df['time'].dt.date
    #df['hour'] = df['time'].dt.hour
    # df = df.drop_duplicates(['ship','month'])
    #df['weekday'] = df['time'].dt.weekday
    df = pd.DataFrame({'lasting_days':[lasting_days]})
    return df




def unique_process(df):
    features_list = ['x', 'y']
    prop = []
    columns = ['x_less_prop', 'x_no_less_prop', 'y_less_prop', 'y_no_less_prop']

    for feature in features_list:
        #temp1 = df[feature].value_counts()
        # 统计数值出现频率小于0.09（出现次数小于约30-40次样本）的占的比例
        temp = df[feature].value_counts(normalize=True)
        #print(temp)
        less_prop = temp[temp < 0.09].sum()
        no_less_prop = 1 - less_prop
        prop.extend([less_prop, no_less_prop])
        # 获取值出现频率大于0.09的那个feature（如x）的值
        #temp2 = (temp1[df[feature].value_counts(normalize=True) < 0.09])
        #print(temp2)
        #print(df[feature].loc[temp2, ])
        #print(temp2)
        #print(temp1)
        #print('=='*22)
    feature_df = pd.DataFrame(dict(zip(columns, prop)), index= [0])
    return feature_df
    #print(feature_df)




def features_engineering(mode= 'train'):
    csvs_dir = train_csvs_dir if mode == 'train' else test_csvs_dir

    fun_list = ['max', 'min', 'mean', 'std', 'skew', 'sum']
    keys = ['ship', 'x', 'y', 'v', 'd']
    i = 0
    df_list = []
    for csv_path in tqdm(glob(csvs_dir)):
        '''
        if i == 2:
            break
        '''
        df = pd.read_csv(csv_path)
        if mode == 'train':
            df.columns = ['ship','x','y','v','d','time','type']
            label = LABEL_DICT[df['type'].unique()[-1]]
        else:
            df.columns = ['ship', 'x', 'y', 'v', 'd', 'time']
        common_features = get_8_values(df, keys= keys, fun_list= fun_list)
        #print(common_features.shape)
        #print(common_features.index)
        # 将x, y, v 划分成10桶
        nums_prop_10 = cut_df_into_10_(df)
        #print(nums_prop_10.index)
        #print(nums_prop_10.shape)
        features_v = process_v(df= df)
        # 获取持续天数
        lasting_days = extract_date_time(df= df)
        x_y_frequency = unique_process(df= df)

        new = pd.concat([common_features, nums_prop_10, features_v, lasting_days, x_y_frequency], axis= 1)
        if mode == 'train':
            new['type'] = label
        #print(new.shape)
        # 判断构造出来的特征的df的columns值是否有重复
        assert ((new.columns.value_counts() == 1).unique()).all() == True, 'columns repeats!!'
        df_list.append(new)

        #i +=1
    dataset_df = pd.concat(df_list, axis= 0)
    print(dataset_df.shape)
    dataset_df.to_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess' + '\\' +mode + '.h5', key= 'df', mode= 'w')
    (dataset_df.reset_index()).to_hdf(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess' + '\\' + mode + '_reset.h5',key= 'df', mode= 'w')






class TEST(object):
    def __init__(self):
        self.path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102\14.csv'


    def test_get_8_values(self):
        df = pd.read_csv(self.path)
        df.columns = ['ship','x','y','v','d','time','type']
        fun_list = ['max', 'min', 'mean', 'std', 'skew', 'sum']
        keys = ['ship', 'x', 'y', 'v', 'd']
        temp = get_8_values(df, keys= keys, fun_list= fun_list)
        print(temp)

    def test_unique_process(self):
        df = pd.read_csv(self.path)
        df.columns = ['ship','x','y','v','d','time','type']
        unique_process(df)

    def test_cut_df_into_10(self):
        df = pd.read_csv(self.path)
        df.columns = ['ship','x','y','v','d','time','type']
        cut_df_into_10_(df)

    def test_process_v(self):
        df = pd.read_csv(self.path)
        df.columns = ['ship','x','y','v','d','time','type']
        process_v(df)
    def test_extract_date_time(self):
        df = pd.read_csv(self.path)
        df.columns = ['ship','x','y','v','d','time','type']
        print(extract_date_time(df))

    @staticmethod
    def test_features_engineering():
        #features_engineering()
        features_engineering(mode= 'test')



if __name__ == '__main__':
    #TEST().test_unique_process()
    #TEST().test_extract_date_time()
    #TEST().test_process_v()
    TEST.test_features_engineering()
    #TEST.test_cut_df_into_10()
    #TEST.test_get_8_values()
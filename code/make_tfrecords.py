import pandas as pd, numpy as np
import tensorflow as tf
import glob
import os

LABEL_DICT = {
    '围网': 0,
    '拖网': 1,
    '刺网': 2
}
NEW = pd.DataFrame({
    'x': 0,
    'y': 0,
    '方向': 0,
    '速度': 0
}, index=[1]
)

def _read_csvs(csvs_root_dir, test = None):
    for file in glob.glob(csvs_root_dir + '\\' + '*.csv'):
        df = pd.read_csv(file)
        if not test:
            label = df['type'].unique()[-1]
            label = LABEL_DICT[label]
        else:
            sequence = df['渔船ID'].unique()[0]

        df = df[['x', 'y', '速度', '方向']]
        df[['x', 'y']] = df[['x', 'y']] / 1.0e+07
        df['方向'] = df['方向'] / 360.
        df['速度'] = df['速度'] / 10.

        index_max = df.index.max()
        #print('index_max', index_max)
        if index_max < 350:
            for i in range(index_max, 349):
                df = df.append(NEW, ignore_index=True, sort= False)
        else:
            df = df.iloc[: 350]
        #print(df.iloc[370:])
        #print(df.to_numpy().shape)
        if not test:
            yield (df , label)
        else:
            yield (df, sequence)

def make_train_tfrecords(generator):
    tfrecords_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\tfrecord' + '\\' + '350_4_3_train.tfrecord'
    if os.path.exists(tfrecords_path):
        os.remove(tfrecords_path)
        print('clean origional tfrecords')

    def _bytes_feature(values):
        return tf.train.Feature(bytes_list= tf.train.BytesList(value= [values]))
    def _int_feature(values):
        return tf.train.Feature(int64_list = tf.train.Int64List(value= [values]))
    writer = tf.io.TFRecordWriter(tfrecords_path)
    ones = np.ones((350, 4, 3))
    for i, da in enumerate(generator):
        data = np.expand_dims(da[0].to_numpy(), 2)
        data = ones * data
        print(data.shape)
        label = da[1]
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'data': _bytes_feature(data.tobytes()),
                    'label': _int_feature(label)
                }
            )
        )
        writer.write(example.SerializeToString())
        '''
        if i % 100 == 0:
            print(i, end= '   ')
            print(label)
        '''
    writer.close()

def make_test_tfrecords(generator):
    tfrecords_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\tfrecord' + '\\' + '350_4_3_test.tfrecord'
    if os.path.exists(tfrecords_path):
        os.remove(tfrecords_path)
        print('clean origional tfrecords')
    def _int_feature(values):
        return tf.train.Feature(int64_list = tf.train.Int64List(value= [values]))
    def _bytes_feature(values):
        return tf.train.Feature(bytes_list= tf.train.BytesList(value= [values]))
    writer = tf.io.TFRecordWriter(tfrecords_path)
    ones = np.ones((350, 4, 3))
    for i, da in enumerate(generator):
        data = np.expand_dims(da[0].to_numpy(), 2)
        data = ones * data
        sequence = da[1]
        print(data.shape)
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'data': _bytes_feature(data.tobytes()),
                    'sequence':_int_feature(sequence)
                }
            )
        )
        writer.write(example.SerializeToString())

        if i % 100 == 0:
            print(i)

    writer.close()

class Test_methond:
    @classmethod
    def _test_read_csv(cls):
        train_root = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102'
        for i, df in enumerate(_read_csvs(train_root)):
            if i < 5:
                print(df.head())
                print('+'*22)
            else:
                break
    @staticmethod
    def _test_df():
        new = pd.DataFrame({
            'x': 0 ,
            'y': 0 ,
            '方向': 0 ,
            '速度': 0
        }, index= [1]
        )
        path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102\472.csv'
        df = pd.read_csv(path)
        df = df[['x', 'y', '速度', '方向']]
        '''
        df[['x', 'y']] = df[['x', 'y']] / 1.0e+07
        df['方向'] = df['方向'] / 360.
        df['速度'] = df['速度'] / 10.
        '''
        index_max = df.index.max()
        print('index_max', index_max)
        if index_max < 400:
            for i in range(index_max, 399):
                df = df.append(new, ignore_index= True)
        else:
            df = df.iloc[: 400]
        #print(df.iloc[370: ])
        print(df.head())
        df = df.to_numpy()
        print(np.expand_dims(df, 2).shape)
        print(df)

    @staticmethod
    def start_make_train_tfrecord():
        csvs_root_dir = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102'
        gengrator = _read_csvs(csvs_root_dir= csvs_root_dir)
        make_train_tfrecords(generator= gengrator)
    @staticmethod
    def _test_generator():
        csvs_root_dir = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102'
        for i, da in enumerate(_read_csvs(csvs_root_dir)):
            if i>0:
                break

            print(da[0].to_numpy().shape)
            print(type(da[1]))
            print('='*22)
    @staticmethod
    def start_make_test_tfrecord():
        csvs_root_dir = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_testA_20200102'
        gengrator = _read_csvs(csvs_root_dir= csvs_root_dir, test= True)
        make_test_tfrecords(generator= gengrator)




if __name__ == '__main__':
    Test_methond.start_make_test_tfrecord()
    #Test_methond.start_make_train_tfrecord()
    #Test_methond._test_df()
    #Test_methond._test_generator()

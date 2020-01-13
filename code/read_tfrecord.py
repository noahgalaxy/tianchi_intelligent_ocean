import tensorflow as tf
import os
import glob
from PIL import Image
import numpy as np, pandas as pd
from matplotlib import pyplot as plt


def read_tfrecords():
    #record_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\tfrecord\400_4_1_train.tfrecord'
    record_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\tfrecord\train.tfrecord'
    #record_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\tfrecord\350_4_3_train.tfrecord'
    reader = tf.data.TFRecordDataset(record_path)
    reader = reader.shuffle(5000).batch(32)
    def _sin(example_poto):
        features = tf.io.parse_example(example_poto, # example_poto[tf.newaxis],
                                              features={
                                                  'data': tf.io.FixedLenFeature([], tf.string),
                                                  'label': tf.io.FixedLenFeature([], tf.int64),
                                              }
                                              )
        label = features['label']
        data = tf.io.decode_raw(features['data'], out_type= tf.float64)
        data = tf.cast(data, tf.float64)
        data = tf.reshape(data, (-1, 400, 4, 3))
        return (data, label)
    for da in reader:
        ds = _sin(da)
        yield ds


def read_test_tfrecords():
    #record_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\tfrecord\_400_4_1_test.tfrecord'
    record_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\tfrecord\test.tfrecord'
    #record_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\tfrecord\350_4_3_test.tfrecord'

    reader = tf.data.TFRecordDataset(record_path)
    reader = reader.shuffle(2000).batch(32)
    def _sin(example_poto):
        features = tf.io.parse_example(example_poto, # example_poto[tf.newaxis],
                                              features={
                                                  'data': tf.io.FixedLenFeature([], tf.string),
                                                  'sequence': tf.io.FixedLenFeature([], tf.int64),
                                              }
                                              )
        sequence = features['sequence']
        data = tf.io.decode_raw(features['data'], out_type= tf.float64)
        data = tf.cast(data, tf.float64)
        data = tf.reshape(data, (-1, 400, 4, 3))
        return (data, sequence)
    for da in reader:
        ds = _sin(da)
        yield ds

def _test():
    ds = read_tfrecords()
    for i, j in enumerate(ds):
        if i > 0:
            break
        print(j[1].shape)
        print(j[0].shape)
        #print(j[0])
        #print(j[1])
    print(i)


def _test_record():
    ds = read_test_tfrecords()
    for i, data in enumerate(ds):
        print(i)
        print(data[0].shape)
        print(data[1])
if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    #_test()
    _test_record()
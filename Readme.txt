# tianchi_intelligent_ocean

1、CONFIG里面是特征列表；

2、code文件夹里面的make_tfrecord和read_tfrecord是制作转换数据格式，制作数据集，Use_cnn是加载数据集并使用cnn训练，predict_save_to_csv加载训练模型，保存预测结果，此四个文件为一套；

3、features_engineering制作特征，train_with_lightgbm使用lightgbm训练，训练时可以在CONFIG里面选择特征，此三个文件为一套；

4、code里面的mlp是使用简单的全连接网络进行训练，效果不好，和cnn一样，准确率0.62左右；

5、tianchi_ship_2019-master为人家的baseline，可忽略；
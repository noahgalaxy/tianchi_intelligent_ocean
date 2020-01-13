import tensorflow as tf
from Use_cnn import MyModel
from read_tfrecord import read_test_tfrecords
import pandas as pd
import numpy as np

LABEL_DICT = {
     0 : '围网',
     1 : '拖网',
     2 : '刺网'
}
model = MyModel()
model.load_weights(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Models\model_weights_40')
DF = pd.DataFrame(columns= ['type'])
for i, data in enumerate(read_test_tfrecords()):

    prediction = model(data[0])
    index = data[1].numpy()
    #print(index)
    fish_type_code = tf.argmax(prediction, 1).numpy()
    #print(fish_type_code)
    fish_type = [LABEL_DICT[x] for x in fish_type_code]
    #print(fish_type)
    df = pd.DataFrame(data= fish_type, index= index, columns= ['type'])
    DF = pd.concat([df, DF], 0)


print('='*22)
print(DF)
DF.sort_index(ascending= True, inplace= True)
print('+'*22)
print(DF)
DF.to_csv(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\Results\result.csv', header= False)

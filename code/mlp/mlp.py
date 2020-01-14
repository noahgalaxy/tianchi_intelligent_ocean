import tensorflow as tf
import pandas as pd, numpy as np
from sklearn import metrics
from CONFIG.CONFIG import FEATURES,FEATURES_1

class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(64, activation= 'relu')
        self.d2 = tf.keras.layers.Dense(128, activation= 'relu')
        #self.drop1 = tf.keras.layers.Dropout(0.4)
        self.d3 = tf.keras.layers.Dense(3, activation= 'softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.d1(x)
        x = self.d2(x)
        #x = self.drop1(x)
        x = self.d3(x)

        return x

def get_dataset():
    test_hdf = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\test_reset.h5'
    train_hdf = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\dataset_after_preprocess\train_reset.h5'

    features = FEATURES.copy()

    for temp in ['ship', 'type']:
        features.remove(temp)

    train_data = pd.read_hdf(train_hdf, key='df')

    test_data = pd.read_hdf(test_hdf, key='df')
    #test_label.drop('index', axis=1, inplace=True)
    # 找出标签和船号
    train_label = tf.cast((train_data['type'].values), tf.float64)
    test_ship = tf.cast(test_data['ship'].values, tf.float64)
    # 去除train_data, test_data里面的'ship'和'type'

    train_data = train_data[features]/ train_data[features].mean()
    train_data = tf.cast(train_data.values, tf.float64)
    #print(train_data.shape)

    test_data = test_data[features] / test_data[features].mean()
    test_data = tf.cast(test_data.values, tf.float64)

    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_label)).shuffle(10000).batch(256)
    test_ds = tf.data.Dataset.from_tensor_slices((test_data)).batch(32)

    return (train_ds, test_ds, test_ship)


def train():
    EPOCHES = 20
    model = MLP()
    model.build(input_shape= (32, 140))
    train_ds, test_ds, test_ship = get_dataset()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    optimizer = tf.keras.optimizers.Adam()
    @tf.function
    def _train_step(batch_data, batch_labels):
        with tf.GradientTape() as tape:
            pred = model(batch_data)
            #loss = metrics.f1_score(batch_labels.numpy(), tf.argmax(pred, axis= 1).numpy(), average='macro')
            loss = loss_object(batch_labels, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(batch_labels, pred)
    for epoch in range(EPOCHES):
        for i, (batch_data, batch_labels) in enumerate(train_ds):
            _train_step(batch_data, batch_labels)

        template = 'Epoch {}, Step{} ,Loss{} , Accuracy: {}'
        print(template.format(epoch + 1,
                                i + 1,
                                train_loss.result(),
                                train_accuracy.result() * 100,
                                )
                )


def tet():
    '''
    model = MLP()
    print(model.summary())
    '''
    train_ds, test_ds, test_ship = get_dataset()
    for batch_data, batch_labels in train_ds.take(1):
        print(batch_data.shape)
        print(batch_labels.numpy())
        print('=='*22)
        #print(model(batch_data))
def h_test_1111():
    get_dataset()

if __name__ == '__main__':
    # 显存按需增长

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    train()

    #tet()
    #h_test_1111()

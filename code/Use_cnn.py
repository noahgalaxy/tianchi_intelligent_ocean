import tensorflow as tf
from read_tfrecord import read_tfrecords


class MyModel(tf.keras.Model):
  def __init__(self, kernel_size= (4, 2), pool_size= (2, 2), features= 32):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(features*1, kernel_size= kernel_size, strides= (1, 1), padding= 'same', activation='relu', kernel_initializer='uniform')
    #self.pool1 = tf.keras.layers.MaxPool2D(pool_size= pool_size, strides= (1, 1), padding= 'same')
    self.pool1_1 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(2, 1), padding='same')
    #self.drop1 = tf.keras.layers.Dropout(0.4)
    # 200x4x32

    self.conv2 = tf.keras.layers.Conv2D(features*2, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')
    #self.pool2 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(1, 1), padding='same')
    self.conv2_1 = tf.keras.layers.Conv2D(features*2, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')
    self.pool2_1 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(2, 1), padding='same')
    #self.drop2 = tf.keras.layers.Dropout(0.4)
    # 100x4x64

    self.conv3 = tf.keras.layers.Conv2D(features*4, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')
    #self.pool3 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(1, 1), padding='same')
    self.conv3_1 = tf.keras.layers.Conv2D(features*4, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')
    self.pool3_1 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(2, 1), padding='same')
    #self.drop3 = tf.keras.layers.Dropout(0.4)
    # 50x4x128
    self.conv4 = tf.keras.layers.Conv2D(features*8, kernel_size=kernel_size, strides=(2, 1), padding='same', activation='relu', kernel_initializer='uniform')
    #self.pool4 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(1, 1), padding='same')
    self.conv4_1 = tf.keras.layers.Conv2D(features*8, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')
    self.pool4_1 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(2, 1), padding='same')
    #self.drop4 = tf.keras.layers.Dropout(0.4)
    # 12x4x256
    self.conv5 = tf.keras.layers.Conv2D(features*16, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')
    self.conv5_1 = tf.keras.layers.Conv2D(features*16, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')
    self.pool5 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=2, padding='same')
    #self.drop4 = tf.keras.layers.Dropout(0.4)
    # 6x2x512
    self.conv6 = tf.keras.layers.Conv2D(features*16, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')
    self.conv6_1 = tf.keras.layers.Conv2D(features*16, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')
    self.pool6 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(2, 1), padding='same')
    # 3x2x512
    self.conv7 = tf.keras.layers.Conv2D(features*16, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')
    self.pool7 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides= 2, padding='same')
    # 1x1x512
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(1024, activation='relu')
    self.drop1_1 = tf.keras.layers.Dropout(0.5)
    self.d2 = tf.keras.layers.Dense(512, activation='relu')
    self.drop2_1 = tf.keras.layers.Dropout(0.5)
    self.d3 = tf.keras.layers.Dense(3, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    #x = self.pool1(x)
    #x = self.conv1_1(x)
    x = self.pool1_1(x)
    #x = self.drop1(x)

    x = self.conv2(x)
    #x = self.pool2(x)
    x = self.conv2_1(x)
    x = self.pool2_1(x)
    #x = self.drop2(x)

    x = self.conv3(x)
    #x = self.pool3(x)
    x = self.conv3_1(x)
    x = self.pool3_1(x)
    #x = self.drop3(x)

    x = self.conv4(x)
    #x = self.pool4(x)
    x = self.conv4_1(x)
    x = self.pool4_1(x)
    #x = self.drop4(x)

    x = self.conv5(x)
    x = self.conv5_1(x)
    x = self.pool5(x)

    x = self.conv6(x)
    x = self.conv6_1(x)
    x = self.pool6(x)

    x = self.conv7(x)
    x = self.pool7(x)

    x = self.flatten(x)

    x = self.d1(x)
    x = self.drop1_1(x)
    x = self.d2(x)
    x = self.drop2_1(x)
    x = self.d3(x)


    return x



def _test1():
    model = MyModel()
    model.build(input_shape= (None, 400, 4, 3))
    print(model.summary())
    print(model.outputs)

def train():
    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    EPOCHS = 25

    for epoch in range(EPOCHS):
        for i, (data, label) in enumerate(read_tfrecords()):
            train_step(data, label)

            template = 'Epoch {}, Step: {},Loss: {}, Accuracy: {}'
            print(template.format(epoch + 1,
                                  i,
                                  train_loss.result(),
                                  train_accuracy.result() * 100
                                  )
                  )
    model.save_weights(r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Models\400_4_3\model_weights_400_4_3_' + str((train_accuracy.result()).numpy()*100))
if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    train()
    #_test1()



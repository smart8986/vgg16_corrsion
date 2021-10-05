import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras.models
from torch.utils.data.dataset import T
resize = 224
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# path = "dogCat/train" #数据集位置
path = "data"  # 数据集位置
train_rate = 0.8

def load_data():
    imgs = [os.path.join(path, img) for img in os.listdir(path)]
    all_data = []
    all_label = []
    for i in range(len(imgs)):
        img = cv2.imread(imgs[i])
        img = cv2.resize(img, (resize, resize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        all_data.append(img)
        if 'uncorrosion' in imgs[i]:
            all_label.append(0)  # 未腐蚀的类别为0
        else:
            all_label.append(1)  # 腐蚀的类别为1

    all_data = np.array(all_data)
    all_label = np.array(all_label)

    # 将数据集分为训练集和测试集
    count = len(imgs)
    train_data = all_data[:int(count*train_rate)]
    train_label = all_label[:int(count*train_rate)]
    test_data = all_data[int(count*train_rate):]
    test_label = all_label[int(count*train_rate):]
    return train_data, train_label, test_data, test_label


def vgg16():
    weight_decay = 0.0005
    nb_epoch = 100
    batch_size = 32

    # layer1
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=(224, 224, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # layer2
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer3
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # layer4
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer5
    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # layer6
    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # layer7
    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer8
    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # layer9
    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # layer10
    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer11
    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # layer12
    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # layer13
    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    # layer14
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # layer15
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # layer16
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    # import data
    train_data, train_label, test_data, test_label = load_data()
    train_label = keras.utils.to_categorical(train_label, 2)  # 把label转成onehot化
    test_label = keras.utils.to_categorical(test_label, 2)  # 把label转成onehot化

    model = vgg16()
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 设置优化器为SGD
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 设置优化器为SGD
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    history = model.fit(train_data, train_label,
                        batch_size=10,
                        epochs=100,
                        validation_split=0.2,  # 把训练集中的五分之一作为验证集
                        shuffle=True)
    model.save('vgg16_corrosion.h5')

    # huatu
    acc = history.history['acc']  # 获取训练集准确性数据
    val_acc = history.history['val_acc']  # 获取验证集准确性数据
    loss = history.history['loss']  # 获取训练集错误值数据
    val_loss = history.history['val_loss']  # 获取验证集错误值数据
    epochs = range(1, len(acc) + 1)
    # 以epochs为横坐标，以训练集准确性为纵坐标
    plt.plot(epochs, acc, 'bo', label='Trainning acc')
    # 以epochs为横坐标，以验证集准确性为纵坐标
    plt.plot(epochs, val_acc, 'b', label='Vaildation acc')
    plt.legend()  # 绘制图例，即标明图中的线段代表何种含义
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:21:31 2019

@author: hanlin
"""

from __future__ import print_function
import numpy as np
import keras
import scipy.io as sio
from keras.callbacks import ModelCheckpoint
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, add, Conv2D, BatchNormalization, Subtract, Dropout,MaxPooling2D,Flatten
from keras.utils import np_utils
from keras import backend as K
batch_size = 32
nb_classes = 10
nb_epoch = 50
# 输入图像的维度，此处是mnist图像，因此是60*60
img_rows, img_cols = 60, 60
# 卷积层中使用的卷积核的个数
nb_filters = 32
# 池化层操作的范围
pool_size = (2,2)
# 卷积核的大小
kernel_size = (3,3)
#
train_dataX = np.load('DeepNIS_pic_train_20dB_module2.npy')
train_dataY = np.load('DeepNIS_class_train_20dB_module2.npy')
test_dataX = np.load('DeepNIS_pic_test_20dB_module2.npy')
test_dataY = np.load('DeepNIS_class_test_20dB_module2.npy')
input_shape = (1,img_rows, img_cols)

# 打印出相关信息
print('train_dataX shape:', train_dataX.shape)
print(train_dataX.shape[0], 'train samples')
print(test_dataX.shape[0], 'test samples')

print('train_dataY shape:', train_dataY.shape)
print(train_dataY.shape[0], 'train samples')
print(test_dataY.shape[0], 'test samples')
# 建立序贯模型
model = Sequential()

# 卷积层
model.add(Conv2D(64, 3, activation='tanh', padding='valid', data_format='channels_first', use_bias=True,
                 input_shape=input_shape ))
# 池化层
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
#
model.add(Conv2D(64, 3, activation='tanh', padding='valid', data_format='channels_first', use_bias=True,
                 ))
# Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())

# 包含128个神经元的全连接层，激活函数为ReLu，dropout比例为0.5
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 包含10个神经元的输出层，激活函数为Softmax
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 输出模型的参数信息
model.summary()
# 配置模型的学习过程
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model_checkpoint = ModelCheckpoint('weights.h5', monitor='loss', save_best_only=True, verbose=1)
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
                                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
LR = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=0, verbose=1, mode='auto',
                                       cooldown=0, min_lr=0)

# 训练模型
model.fit(train_dataX, train_dataY, batch_size=batch_size, nb_epoch=nb_epoch,shuffle=True,
          verbose=1, validation_data=(test_dataX, test_dataY),
          callbacks=[model_checkpoint, tensorboard, LR])

# 按batch计算在某些输入数据上模型的误差
score = model.evaluate(test_dataX, test_dataY, verbose=0)
# 输出训练好的模型在测试集上的表现
print('Test score:', score[0])
print('Test accuracy:', score[1])

testY_predict = model.predict(test_dataX, verbose=1)
sio.savemat('test_result_DeepNIS_20dB_module2.mat', { 'testY': test_dataY, 'testY_predict': testY_predict})


trainY_predict = model.predict(train_dataX, verbose=1)
sio.savemat('train_result_DeepNIS_20dB_module2.mat', { 'trainY': train_dataY, 'trainY_predict': trainY_predict})




























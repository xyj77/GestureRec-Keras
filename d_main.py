# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:10:25 2019

@author: xyj77
"""

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import optimizers
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from model.LeNet import *
from model.resnet import *
from model.Vgg import *
from model.DenseNet import *

from keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

# Hyper-parameter
DATA_PATH = './data'
LOG_PATH = './log'
IMG_SIZE = 32
CLASS = 9
BATCH_SIZE = 32
ITERATION = 20

# MODEL = 'LeNet'
# MODEL = 'VGG'
MODEL = 'ResNet'
# MODEL = 'DenseNet'
# MODEL = 'SENet'


# build network
if MODEL == 'LeNet':
    EPOCH = 500
    model = LeNet(in_shape=(IMG_SIZE,IMG_SIZE,1), n_class=CLASS)
elif MODEL == 'VGG':
    EPOCH = 200
    model = Vgg19(in_shape=(IMG_SIZE,IMG_SIZE,1), n_class=CLASS)
elif MODEL == 'ResNet':
    EPOCH = 400
    model = ResnetBuilder.build_resnet_18((1, IMG_SIZE, IMG_SIZE), CLASS)
elif MODEL == 'DenseNet':
    EPOCH = 200
    model = DenseNet(in_shape=(IMG_SIZE,IMG_SIZE,1), n_class=CLASS)
elif MODEL == 'SENet':
    EPOCH = 200
    model = DenseNet(in_shape=(IMG_SIZE,IMG_SIZE,1), n_class=CLASS)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# show model
plot_model(model, to_file='./images/'+ MODEL +'_model.png')
print(model.summary())

# set callback
tb_cb = TensorBoard(log_dir=LOG_PATH)
cbks = [tb_cb]

# using real-time data augmentation
print('Using real-time data augmentation.')
datagen = ImageDataGenerator(
        rescale=1./255,
        # featurewise_center=True,
        rotation_range=30,
        horizontal_flip=True,
        validation_split=0.2)

train = datagen.flow_from_directory(
        DATA_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training')
                    
validation = datagen.flow_from_directory(
        DATA_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation')

# start train 
print('Using ' + MODEL + 'to predict gestures!')
history = model.fit_generator(
        generator=train,
        steps_per_epoch=ITERATION,
        epochs=EPOCH,
        callbacks=cbks,
        validation_data=validation,
        validation_steps=ITERATION)

# save model
model.save('./model/'+ MODEL +'_model.h5')

# visualization
# Acc
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.savefig('./images/'+ MODEL +'_acc.png')
plt.clf()

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.savefig('./images/'+ MODEL +'_loss.png')
plt.close()

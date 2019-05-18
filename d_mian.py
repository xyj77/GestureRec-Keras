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
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from model.LeNet import *
from model.resnet import *

# Hyper-parameter
DATA_PATH = './data'
LOG_PATH = './log'
IMG_SIZE = 32
CLASS = 3
EPOCH = 100
BATCH_SIZE = 16
ITERATION = 20
MODEL = 'LeNet'


# build network
if MODEL == 'LeNet':
    # LeNet
    model = LeNet(in_shape=(IMG_SIZE,IMG_SIZE,1), n_class=CLASS)
elif MODEL == 'ResNet':
    # Resnet
    EPOCH = 60
    model = ResnetBuilder.build_resnet_18((1, IMG_SIZE, IMG_SIZE), CLASS)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

plot_model(model, to_file='./images/'+ MODEL +'_model.png')
print(model.summary())

# set callback
def scheduler(epoch):
    if epoch < 50:
        return 0.01
    if epoch < 100:
        return 0.005
    return 0.001
tb_cb = TensorBoard(log_dir=LOG_PATH)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr, tb_cb]

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
history = model.fit_generator(
        generator=train,
        steps_per_epoch=ITERATION,
        epochs=EPOCH,
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


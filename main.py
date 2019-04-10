#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:36:37 2019

@author: lierlong
"""

from keras.utils import Sequence
from keras import backend as K
#import os
import numpy as np
import random
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD 


img_shape = (128, 64, 3) 
pic_path = './database/spectrograms/'
#names = [name for name in os.listdir(pic_path) if name.split('.')[-1]=='png']

def name_generate():
    train_pic = []
    val_pic = []
    names = ['_jackson_','_nicolas_','_theo_','_yweweler_']
    for i in range(9):
        for name in names:
            num = list(range(50))
            random.shuffle(num)
            train_pic += [str(i) + name + str(j) + '.png' for j in num[:35]]
            val_pic += [str(i) + name + str(j)+ '.png' for j in num[35:]]
    random.shuffle(train_pic)
    random.shuffle(val_pic)
    return train_pic,val_pic
train_pic, val_pic = name_generate()


class TrainingData(Sequence):
    def __init__(self, data, steps=1000, batch_size=32):
        """
        """
        super(TrainingData, self).__init__()
        self.steps = steps
        self.batch_size = batch_size
        self.data = data

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.data))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        c = np.zeros((size, 4), dtype=K.floatx())
        for i in range(size):
            a[i, :, :, :] = self.read_pic(start + i)
            label = self.data[start + i].split('_')[1]
            if label == 'nicolas':c[i,0] = 1
            elif label == 'jackson':c[i,1] = 1
            elif label == 'theo':c[i,2] = 1
            else:c[i,3] = 1
        return a, c
    
    def read_pic(self,i):
        img = Image.open(pic_path + self.data[i])
        img = img_to_array(img)
        img -= np.mean(img, keepdims=True)
        img /= np.std(img, keepdims=True) + K.epsilon()
        return img

    def __len__(self):
        return len(self.data) // self.batch_size

def VGG_16():
    model = Sequential()

    # BLOCK 1
    model.add(Conv2D(filters= 32, kernel_size=(3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1', input_shape = (224, 224,3)))   
    model.add(Conv2D(filters= 32, kernel_size=(3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block1_pool'))
 
    # BLOCK2
    model.add(Conv2D(filters= 64, kernel_size=(3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1'))   
    model.add(Conv2D(filters= 64, kernel_size=(3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block2_pool'))
 
    # BLOCK3
    model.add(Conv2D(filters= 128, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2'))
    model.add(Conv2D(filters= 128, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv3'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block3_pool'))
 
    # BLOCK4
    model.add(Conv2D(filters= 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2'))
    model.add(Conv2D(filters= 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv3'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block4_pool'))
 
    # BLOCK5
    model.add(Conv2D(filters= 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv2'))
    model.add(Conv2D(filters= 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv3'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block5_pool'))

    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu', name = 'fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation = 'softmax', name = 'prediction'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr =0.0001 ),
                  metrics=['categorical_crossentropy','accuracy'])
    
    return model
















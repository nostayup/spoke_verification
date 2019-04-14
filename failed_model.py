#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:28:22 2019

@author: lierlong
"""


def VGG_16():
    '''
    这个模型训练250步只能测试集到0.58
    并且增加层数会使性能下降
    '''
    model = Sequential()
    # BLOCK 1
    model.add(Conv2D(filters= 64, kernel_size=(3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1', input_shape = (128, 64,3)))   
    model.add(Conv2D(filters= 64, kernel_size=(3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block1_pool'))

    # BLOCK2
    model.add(Conv2D(filters= 128, kernel_size=(3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1'))   
    model.add(Conv2D(filters= 128, kernel_size=(3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block2_pool'))
 
    # BLOCK3
    model.add(Conv2D(filters= 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2'))
    model.add(Conv2D(filters= 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv3'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block3_pool'))
 
    # BLOCK4
    model.add(Conv2D(filters= 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2'))
    model.add(Conv2D(filters= 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv3'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block4_pool'))
 
    # BLOCK5
#    model.add(Conv2D(filters= 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv2'))
#    model.add(Conv2D(filters= 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv3'))
#    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block5_pool'))

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu', name = 'fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation = 'softmax', name = 'prediction'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr =0.0001 ),
                  metrics=['categorical_crossentropy','accuracy'])
    return model

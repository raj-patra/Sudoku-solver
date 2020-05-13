# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:58:28 2019

Creating the Model for training the Sudoku-CNN

@author: Raj Kishore Patra
"""
import keras
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape

def train_mod(x_train, y_train, batch_size=64, ep=2):
    
    print('\n*****Training the model*****\n')
    
    model = keras.models.Sequential()
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    
    model.add(Flatten())
    
    model.add(Dense(81*9))
    model.add(Reshape((-1,9)))
    model.add(Activation('softmax'))
    
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=ep)
    model.summary()
    
    model.save('sudoku.model')
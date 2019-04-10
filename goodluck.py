# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:45:19 2019

@author: lierl
"""

import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
from keras import backend as K
import numpy as np


audio_path = './database/recordings/'
names = os.listdir(audio_path)

#显示波形
def waveplot(i):
    x , sr = librosa.load(audio_path + names[0])
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
#显示频谱
def stftplot(i):
    x , sr = librosa.load(audio_path + names[i])
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(7, 3))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

def melplot(i):
    x , sr = librosa.load(audio_path + names[i])
    X = librosa.feature.melspectrogram(x, n_mels=128,hop_length = 256)
    print(X.shape)
    Xdb = librosa.amplitude_to_db(abs(X))
    print(Xdb.shape)
    plt.figure(figsize=(7, 3))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
#    plt.imshow(Xdb)
    
def savepic_mel(i):
    x , sr = librosa.load(audio_path + names[i])
    X = librosa.feature.melspectrogram(x, n_mels=128,hop_length = 256)
    print(X.shape)
    Xdb = librosa.amplitude_to_db(abs(X))
    print(Xdb.shape)
    plt.figure(figsize=(7, 3))
    Xdb -= np.mean(Xdb, keepdims=True)
    Xdb /= np.std(Xdb, keepdims=True) + K.epsilon()
    plt.imsave('./spectrograms/'+ names[i].split('.')[0] + '.jpg',Xdb)




#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
#plt.colorbar()


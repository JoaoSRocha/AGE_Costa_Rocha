#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 22:01:44 2018

@author: togepi
"""
import numpy as np
import pickle
import os
from scipy.fftpack import fft

TRAIN_DATA_PATH = 'train_data_segments/'

def extract_features(sequence):
    t = sequence[:,0]
    x = sequence[:,1]
    y = sequence[:,2]
    z = sequence[:,3]
    x_fft = fft(x)
    y_fft = fft(y)
    z_fft = fft(z)
    feat_x = np.concatenate((np.real(x_fft[0:150]), np.imag(x_fft[0:150])))
    feat_y = np.concatenate((np.real(y_fft[0:150]), np.imag(y_fft[0:150])))
    feat_z = np.concatenate((np.real(z_fft[0:150]), np.imag(z_fft[0:150])))
    return np.concatenate((feat_x,feat_y,feat_z))


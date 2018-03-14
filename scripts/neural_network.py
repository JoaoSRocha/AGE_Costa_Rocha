#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:06:28 2018

@author: togepi
"""


import AGE_module
import os
import numpy as np
import pickle

os.chdir(AGE_module.ROOT_PATH)

X,y = AGE_module.get_train_feats(save=True)

#%%    
from sklearn.neural_network import MLPClassifier

neural = MLPClassifier(hidden_layer_sizes=(900,500,500,400), verbose=True)
neural.fit(X,y)

#%%
os.chdir(AGE_module.ROOT_PATH)
AGE_module.create_submission_csv(neural)
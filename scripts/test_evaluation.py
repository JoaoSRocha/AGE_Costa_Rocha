#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:14:35 2018

@author: togepi
"""

import AGEmodule as AGE
import numpy as np
import pickle

def calculate_test_features():
    test_feats = []
    test_data = AGE.load_test()
    session_counter = 0
    session_total = len(test_data)
    duration_total = 0
    for session in test_data:
        print('session ', session_counter, session_total)
        session_counter+=1
        walk_segments = AGE.walk_detection(session)
        print('extracted segments: ',len(walk_segments))
        duration_total += np.sum([len(seg[0])/100 for seg in walk_segments])
        session_feats = []
        for segment in walk_segments:
            processed_seg = AGE.preprocess_walk_segment(segment)
            seg_windows = AGE.divide_walk_segment(processed_seg)
            for window in seg_windows:
                window_feats = AGE.window_feature_extraction(window)
                session_feats.append(window_feats)
        test_feats.append(np.array(session_feats))
                
    print('total duration: ', duration_total)

def load_test_features():
    with open(AGE.PATH + 'test_features.pkl', 'rb') as f:
        test_feats = pickle.load(f)
    return test_feats

#%% Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, RFECV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

feature_processing = RobustScaler()

svm_clf = SVC(max_iter = -1, probability = True, kernel='rbf')

pipe = Pipeline([('scale', feature_processing), 
                 ('classifier', RFECV(estimator = svm_clf))])

param_grid = dict(classifier__estimator__C = [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  classifier__estimator__gamma = [ 0.01, 0.001, 0.0001])

tuned_pipeline = GridSearchCV(estimator = pipe, param_grid=param_grid, 
                              cv = 5, verbose = 10000, n_jobs = -1)

#%% Pipeline training
Xtrain, Ytrain = AGE.load_features_and_labels()

tuned_pipeline.fit(Xtrain, Ytrain)


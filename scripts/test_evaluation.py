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
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

feature_processing = RobustScaler()
feature_selection = SelectPercentile(mutual_info_classif, percentile = 45)
svm_clf = SVC(max_iter = -1, probability = True, kernel='rbf', C = 10, gamma = 0.001)

pipe = Pipeline([('scale', feature_processing), 
                 ('selection', feature_selection),
                 ('classifier', svm_clf)])

#%% Pipeline training
Xtrain, Ytrain = AGE.load_features_and_labels()

pipe.fit(Xtrain, Ytrain)
print('training done')
#%%
Xtest = load_test_features()
predictions = np.zeros((26,10))

for i in range(10):
    session = Xtest[i]
    n_samples = len(session)
    sample_probs = np.zeros((26,n_samples))
    for j in range(n_samples):
        sample_probs[:,j] = pipe.predict_proba(session[j,:].reshape(1,-1))
    mean_prob = np.mean(sample_probs, axis=1)
    norm_prob = mean_prob/np.sum(mean_prob)
    predictions[:,i] = norm_prob

predictions_mean = np.mean(predictions, axis=1)
print(predictions_mean)



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

if __name__ =="__main__":
    feature_processing = RobustScaler()

    svm_clf = SVC(max_iter = -1, probability = True, kernel='rbf')

    from tempfile import mkdtemp
    cachedir = mkdtemp()

    # pipe = Pipeline([('scale', feature_processing),
    #                  ('selection',feature_selection),
    #                  ('classifier', svm_clf)],
    #                 memory = cachedir)

    #%% Pipeline training
    Xtrain, Ytrain = AGE.load_features_and_labels()

    Xtrain_processed = feature_processing.fit_transform(Xtrain,Ytrain)

    params = []
    scores = []
    for perc in [10,15,20,25,30, 35, 40, 45, 50]:
        print(perc, ' percent')
        feature_selection = SelectPercentile(mutual_info_classif, percentile=perc)
        Xtrain_selected = feature_selection.fit_transform(Xtrain_processed, Ytrain)

        param_grid = dict(C = [ 10, 100, 1000,10000],
                          gamma = [ 0.01, 0.001, 0.0001])

        tuned_pipeline = GridSearchCV(estimator = svm_clf, param_grid=param_grid,
                                      cv = 3, verbose = 10000,n_jobs=-1)

        tuned_pipeline.fit(Xtrain_selected,Ytrain)

        print('Best parameters for ', perc, '%: ', tuned_pipeline.best_params_)
        print('Best score for  ', perc, '%: ', tuned_pipeline.best_score_)
        params.append(tuned_pipeline.best_params_)
        scores.append(tuned_pipeline.best_score_)

    print(scores)




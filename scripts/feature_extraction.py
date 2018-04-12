#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 19:34:33 2018

@author: togepi
"""

import AGEmodule as AGE
import pickle as pkl
import numpy as np

user_total = AGE.USER_NUMBER
feats = []
labels = []
d_total = np.zeros(user_total+1)

for user_counter in range(1, user_total+1):
    print('user ', user_counter, user_total)
    
    user_data = AGE.load_user(user_counter)
    session_counter = 0
    session_total = len(user_data)
    duration_total = 0
    for session in user_data:
        print('session ', session_counter, session_total)
        session_counter+=1
        walk_segments = AGE.walk_detection(session)
        print('extracted segments: ',len(walk_segments))
        duration_total += np.sum([len(seg[0])/100 for seg in walk_segments])
        
        for segment in walk_segments:
            processed_seg = AGE.preprocess_walk_segment(segment)
            seg_windows = AGE.divide_walk_segment(processed_seg)
            for window in seg_windows:
                window_feats = AGE.window_feature_extraction(window)
                feats.append(window_feats)
                labels.append(user_counter)
                
    print('total duration: ', duration_total)
    d_total[user_counter] = duration_total

feats = np.array(feats)
labels = np.array(labels)
print('extraction done')

with open(AGE.PATH + 'features_09_04.pkl', 'wb') as f:
    pkl.dump(feats,f,-1)
with open(AGE.PATH + 'labels_09_04.pkl', 'wb') as f:
    pkl.dump(labels,f,-1)
print('saved')
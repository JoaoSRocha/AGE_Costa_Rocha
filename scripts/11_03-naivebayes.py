#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 01:20:19 2018

@author: togepi
"""
import numpy as np
import pickle

with open('feature_matrix.pkl','rb') as f:
    feats = np.array(pickle.load(f))
with open('labels.pkl','rb') as f:
    labels = np.array(pickle.load(f))
    
#%%
from sklearn.preprocessing import StandardScaler

feats = StandardScaler().fit_transform(feats)

#%%
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(feats,labels)
#%%
import os

import pandas as pd

questions = pd.read_csv('test/questions.csv')

#%%
from create_feature_matrix import extract_features

classes = classifier.classes_
submission = []
for index, row in questions.iterrows():
    ID = row['QuestionId']
    print(ID)
    sequenceID = row['SequenceId']
    quiz_device = row['QuizDevice']
    with open('test_pickle/'+str(sequenceID)+'.pkl', 'rb') as pkl:
        sequence = pickle.load(pkl)
    feats = extract_features(sequence)
    feats = feats.reshape(1,-1)
    probs = classifier.predict_proba(feats)
    prob_device = float(probs[0,np.argwhere(classes==quiz_device)])
    submission.append((ID,prob_device))
    
#%%
df = pd.DataFrame(data=submission, columns=['QuestionId', 'IsTrue'])
df.to_csv('submission_' + 11_03_2018 +'.csv', index=False)

    
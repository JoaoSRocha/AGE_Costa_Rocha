#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:31:30 2018

@author: togepi

Module for all the useful functions for this project

"""
# Global imports:
import numpy as np
import os
import pickle
from datetime import datetime

# Global Variables (paths):
ROOT_PATH = '/home/togepi/feup-projects/AGE_Costa_Rocha/'
TRAIN_DATA_PATH = 'pickled_train_data_segments/'
TEST_DATA_PATH = 'pickled_test_data/'
SCRIPTS_PATH = 'scripts/'

#%% extract_features function

def extract_features(sequence):
    # extract_features returns the used features for an input 
    # train/test sequence. This function can be adjusted in the future to
    # extract different features
    # Returns the features in a single row vector
    
    # extracts the FFT of each axis, using real and imaginary part of the 
    # first half of the FFT.
    # size = 900 features per sequence
    
    axis = sequence[:,1:4]
    ft = np.fft.fftn(axis)
    ft = ft.flatten()
    feats = np.log(np.abs(ft)+1)
    return feats

#%% get_train_feats

from sklearn.preprocessing import StandardScaler

def get_train_feats(save):
    # get_train_feats creates a feature matrix and label vector using the
    # extract_features function
    # save is a bool that determines if the matrices are pickled and saved
    
    os.chdir(ROOT_PATH)
    device_files = os.listdir(TRAIN_DATA_PATH)
    total_files = len(device_files)
    X = []
    y = []
    i = 0
    # read all the training files
    for file in device_files:
        i+=1
        print(i,'/',total_files)
        # open and load each file
        with open(TRAIN_DATA_PATH + file, 'rb') as pkl:
            data = pickle.load(pkl)
        # each file has a list of 300 training segments
        # and we extract the features for each
        for sequence in data:
            f = extract_features(sequence)
            label = int(sequence[0,-1])
            # add the features and labels to the corresponding matrices
            X.append(f)
            y.append(label)
    
    # features are scaled and standarized
    X = StandardScaler().fit_transform(X)
    
    # if the user decices to save, both the feature and label matrices are
    # pickled and saved using the current time as an unique identifier
    if (save):
        date = datetime.now()
        date = date.strftime('%Y-%m-%d_%H-%M-%S') 
        with open('features_'+date+'.pkl', 'wb') as out:
            pickle.dump(X, out, -1)
        with open('labels_'+date+'.pkl', 'wb') as out:
            pickle.dump(y, out, -1)
        print('Matrices saved')
        
    # and return the matrices
    return X,y
#%% create_submission_csv

import pandas as pd

def create_submission_csv(trained_classifier):
    # create_submission_csv creates a .csv file used for scoring in Kaggle
    # Input: trained_classifier is an already trained classifier with
    # the method predict_proba()
    
    os.chdir(ROOT_PATH)
    # read the questions file
    questions = pd.read_csv('test_questions/questions.csv')
    
    # labels defined by the classifier
    classes = trained_classifier.classes_
    submission = []
    print('Predictiong probabilities of test set:')
    # iterate through all the questions
    for index, row in questions.iterrows():
        # determine question ID
        ID = row['QuestionId']
        print(ID,'/90024')
        
        # determine the sequence to evaluate and the proposed device
        sequenceID = row['SequenceId']
        quiz_device = row['QuizDevice']
        
        # open the test sequence .pkl file
        with open('pickled_test_data/'+str(sequenceID)+'.pkl', 'rb') as pkl:
            sequence = pickle.load(pkl)
        # and extract the corresponding features
        feats = extract_features(sequence)
        
        # reshape as needed to predict probability for every class
        feats = feats.reshape(1,-1)
        probs = trained_classifier.predict_proba(feats)
        
        # probs gives us a 1x384 vector with the probability for each class
        # using the proposed device ID, we only store the probability for said
        # device
        prob_device = float(probs[0,np.argwhere(classes==quiz_device)])
        print(probs)
        # and add it to the results
        submission.append((ID,prob_device))
    
    # convert the submission list to a pandas DataFrame
    df = pd.DataFrame(data=submission, columns=['QuestionId', 'IsTrue'])
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # and save it to .csv
    df.to_csv('submissions/submission_'+date+'.csv', index=False)
    
    print('File Created')
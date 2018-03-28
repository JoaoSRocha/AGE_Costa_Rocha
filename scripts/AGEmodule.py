# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pickle as P
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft

PATH = '/home/togepi/feup-projects/AGE_Costa_Rocha/'
PICKLE_PATH = '/home/togepi/feup-projects/AGE_Costa_Rocha/pickled_data/'
DATASET_PATH = '/home/togepi/feup-projects/AGE_Costa_Rocha/Trainset/'

def create_pickles():
    # creates .pkl files for each user in the dataset
    # each user has a list of sessions
    # each session is divided into each sensor readings
    # sensor readings are a numpy array with timestamp and readings
    
    # gets all users
    user_list = os.listdir(DATASET_PATH)
    user_list.remove('readme.txt')
    
    #columns of the .txt files 
    colnames = ['sensor','timestamp','reading_x', 'reading_y', 'reading_z', 'discard']
    
    # iterate through each user
    for user in user_list:
        print(user)
        # gets all sessions of a user
        session_list = os.listdir(DATASET_PATH+user)
        
        #store here data to pickle
        toPickle = []
        
        #iterate through each user session
        for session in session_list:
            print(session)
            path = DATASET_PATH + user + '/' + session + '/all.txt'
            
            # read the file
            data = pd.read_csv(path, names = colnames)
            # determine rows with different sensors
            bool_acc = data['sensor'] == 'A'
            bool_gyro = data['sensor'] == 'G'
            bool_mag =data['sensor'] == 'M'
            # get those rows
            acc = data[bool_acc].drop(['sensor','discard'], axis=1).as_matrix()
            gyro = data[bool_gyro].drop(['sensor','discard'], axis=1).as_matrix()
            mag = data[bool_mag].drop(['sensor','discard'], axis=1).as_matrix()
            # append to final variable as a tuple
            toPickle.append((acc,gyro,mag))
            print('session done')
        
        #create the pickle file
        with open(PICKLE_PATH+user+'.pkl','wb') as outfile:
            P.dump(toPickle, outfile, -1)
        # counter
        print('user done')
        
def plot_session(sessionData, mag=False):
    # plots in 3 subs the different sensors
    # plotMag is a bool: if true, plots the magnitude of each sensor
    plt.figure()
    if mag:
        ax1=plt.subplot(3,1,1)
        Ma = get_mag(sessionData[0])
        plt.plot(Ma[:,0], Ma[:,1])
        plt.title('Acceleration')
        
        plt.subplot(3,1,2,sharex=ax1)
        Mg = get_mag(sessionData[1])
        plt.plot(Mg[:,0], Mg[:,1])
        plt.title('Gyroscope')
        
        plt.subplot(3,1,3, sharex=ax1)
        Mm = get_mag(sessionData[2])
        plt.plot(Mm[:,0], Mm[:,1])
        plt.title('Magnetometer')
    else:
        ax1=plt.subplot(3,1,1)
        plt.plot(sessionData[0][:,0], sessionData[0][:,1:])
        plt.title('Acceleration')
        
        plt.subplot(3,1,2,sharex=ax1)
        plt.plot(sessionData[1][:,0], sessionData[1][:,1:])
        plt.title('Gyroscope')
        
        plt.subplot(3,1,3, sharex=ax1)
        plt.plot(sessionData[2][:,0], sessionData[2][:,1:])
        plt.title('Magnetometer')

def get_mag(sensorMatrix):
    # calculates the magnitude of sensor readings
    # returns a matrix: first column timestamps, second magnitude
    # sensorMatrix must be of type:
    # timestamp, readings_x, readings_y, readings_z
    # works for all types of sensor
    
    ts = sensorMatrix[:,0]
    x = sensorMatrix[:,1]
    y = sensorMatrix[:,2]
    z = sensorMatrix[:,3]
    mag = np.sqrt(x**2 + y**2 + z**2)
    
    tmag = np.transpose(np.stack((ts,mag)))
    return tmag

def load_dev():
    #15 segundos pe
    #15 sentado
    #15 andar
    #15 movimentos aleatorios
    with open(PATH+'dev.pkl', 'rb') as f:
        dev = P.load(f)
    return dev[0]

def walk_detection(sessionData):
    
    # get gyro data
    gyroMag = get_mag(sessionData[1])
    # convert timestamps to seconds
    gyroMag[:,0] /= 1000000000
    
    # get the mean sampling frequency from total time/samples
    nSamples = np.shape(gyroMag)[0]
    dt = gyroMag[-1,0] - gyroMag[0,0]
    sampling_period = dt/nSamples
    sampling_freq = 1/sampling_period
    
    # calculate the stft:
    # the length of the window is about .7 seconds
    WINDOW_LENGTH = 0.7
    # calculate the number of samples for that:
    N = WINDOW_LENGTH // sampling_period + 1 #+1 to ensure it is at least .7sec
    
    shortFT = stft(gyroMag[:,1], fs=sampling_freq, nperseg=N)
    shortFT_freqs = shortFT[0] #the calculated frequencies
    shortFT_times = shortFT[1] #the time intervals
    shortFT_spect = shortFT[2] #the overall spectrum
    
    # calculate the energy of the stft
    shortFT_energy = np.abs(shortFT_spect)**2
    
    # only consider intervals with relevant frequencies for walking
    # in this case fmax = 7Hz
    F_MAX = 7
    freqsToSum = shortFT_freqs < F_MAX
    
    # sum energy of components up to fmax
    lowEnergy = np.sum(shortFT_energy[freqsToSum], axis=0)
    
    # low componentes tend to present >0 values for walking
    # however, too large values indicate non-walking activity
    # find the values between 2 thresholds:
    THRESH_MIN_ENERGY = .5
    THRESH_MAX_ENERGY = 8
    a = lowEnergy > THRESH_MIN_ENERGY
    b = lowEnergy < THRESH_MAX_ENERGY
    mask = a*b
    
    # necessity to post process the mask so:
    # single rejected points are accepted
    # only segments longer than THRESH_MIN_LENGTH are accepted
    
    return lowEnergy, mask #for now
    
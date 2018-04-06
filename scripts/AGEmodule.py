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
from sklearn.cluster import KMeans
from collections import Counter

PATH = '/home/togepi/feup-projects/AGE_Costa_Rocha/'
PICKLE_PATH = '/home/togepi/feup-projects/AGE_Costa_Rocha/pickled_data/'
DATASET_PATH = '/home/togepi/feup-projects/AGE_Costa_Rocha/Trainset/'
TESTSET_PATH = '/home/togepi/feup-projects/AGE_Costa_Rocha/Testset/'
TESTSET_PICKLE_PATH = '/home/togepi/feup-projects/AGE_Costa_Rocha/test_pickled_data/'

def create_pickles(dataset_path):
    # creates .pkl files for each user in the dataset
    # each user has a list of sessions
    # each session is divided into each sensor readings
    # sensor readings are a numpy array with timestamp and readings
    
    # gets all users
    user_list = os.listdir(dataset_path)
    try:
        user_list.remove('readme.txt')
    except ValueError:
        pass
    
    #columns of the .txt files 
    colnames = ['sensor','timestamp','reading_x', 'reading_y', 'reading_z', 'discard']
    
    # iterate through each user
    for user in user_list:
        print(user)
        # gets all sessions of a user
        session_list = os.listdir(dataset_path+user)
        session_list.sort(key = lambda f: int(f))
        
        #store here data to pickle
        toPickle = []
        
        #iterate through each user session
        for session in session_list:
            print(session)
            path = dataset_path + user + '/' + session + '/all.txt'
            
            # read the file
            data = pd.read_csv(path, names = colnames)
            # subtract first timestamp to all
            min_timestamp = data['timestamp'].min()
            data['timestamp'] = data['timestamp'].sub(min_timestamp)
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
        with open(TESTSET_PICKLE_PATH+user+'.pkl','wb') as outfile:
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

def load_test():
    with open(TESTSET_PICKLE_PATH + 'idX.pkl','rb') as f:
        test = P.load(f)
    return test

def load_user(userID):
    # user ID is just a number
    file = 'id'+str(userID)+'.pkl'
    filepath = PICKLE_PATH + file
    with open(filepath, 'rb') as f:
        data = P.load(f)
    return data

def walk_detection(sessionData, N = 5, neig_size = 9, thresh_max_frequency = 7,
            thresh_min_energy=.5, thresh_max_energy=7, thresh_min_duration = 10, 
            plot=False):
    # detects walk segments
    # returns a list of tuples with the sensor info for each walking segment
    # parameters:
    # N - number of clusters for KMeans
    # neig_size - neighborhood to consider to smooth clustering labels
    # thresh_max_frequency - max frequency to consider for walking energy consideration
    # thresh_min_energy - min energy of low frequency to consider walking
    # thresh_max_energy - max energy of low frequency to consider walking
    # thresh_min_duration - min duration in seconds for a segment to be considered
    # plot - plots the results of the segmentation
    
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
    window_samples = WINDOW_LENGTH // sampling_period + 1 #+1 to ensure it is at least .7sec

    shortFT = stft(gyroMag[:,1], fs=sampling_freq, nperseg=window_samples)
    shortFT_freqs = shortFT[0] #the calculated frequencies
    shortFT_times = shortFT[1] #the time intervals
    shortFT_spect = shortFT[2] #the overall spectrum
    
    # calculate the energy of the stft
    shortFT_energy = np.abs(shortFT_spect)**2
    
    # as is common when dealing with power spectra
    # consider the log of the energy of the signal    
    log_energy = np.log(shortFT_energy)
    
    # apply a clustering method such as KMeans, with N clusters:
    # the goal here is to identify several different "modes" present in
    # the STFT data:
    # - Rest, when the user is not moving => low overall energy
    # - Walking => high energy content in low frequencies, associated with
    #              walking, up to thresh_max_frequency
    # - Noise, when there is a lot of random movement, such as when picking up
    #   the phone, etc => highest energy content in all frequencies
    
    # KMeans will cluster all the time intervals of the STFT according to their
    # similarity in frequency content
    # N is variable, but 5 clusters seem to give good results
    clust = KMeans(n_clusters=N, n_jobs=-1)
    # cluster using the log of the energy
    labels = clust.fit_predict(log_energy.T)

    # The resulting cluster centers
    # Each one is an "average" behaviour present in the signal
    centers = clust.cluster_centers_
    centers = centers.T
    
    # Calculate the energy content of low frequencies
    # first of all, identify which frequency components to sum, according to
    # thresh_max_frequencies
    freqsToSum = shortFT_freqs < thresh_max_frequency
    
    # the cluster centers are calculated with the energy log, so they
    # are exp to recover the frequency content
    centers_exp = np.exp(centers)
    
    # the energy content in low frequencies is calculated for each cluster center
    centers_low_energy = np.sum(centers_exp[freqsToSum], axis=0)
    
    # each cluster can be of three types: static, walking or noisy
    # - static clusters are the ones with energy content lower than thresh_min_energy
    # - walking clusters are between the min and max energy thresholds
    # - noisy clusters have higher energy than max threshold
    centers_static = centers_low_energy<thresh_min_energy
    centers_walk = np.logical_and(centers_low_energy>thresh_min_energy,
                                  centers_low_energy<thresh_max_energy)
    
    # With N clusters, we have N labels. Reduce this number to the 3 meaninful 
    # ones, by identifying the low frequency energy content of each and 
    # compare it to the defined thresholds
    # save these new labels in another vector
    reduced_labels = np.zeros(np.shape(labels))
    
    # go through all the labels
    for i in range(len(labels)):
        label = labels[i]
        
        # and replace according to the center behaviour
        if centers_static[label]:
            # static
            reduced_labels[i] = 0
        elif centers_walk[label]:
            # walking
            reduced_labels[i] = 1
        else:
            # noisy
            reduced_labels[i] = 2
    
    # To remove peaks in the labels (a single interval different from the rest
    # in a sequence), a "filter" is passed through the label vector:
    limits = neig_size//2
    for i in range(len(reduced_labels)):
        # consider a neighborhood defined by the user
        neig = reduced_labels[i-limits:i+limits+1]
        
        # count the labels in the neighborhood and their frequency
        C = Counter(neig)
        try:
            # if one label is most frequent, with more than half
            # of the size of the neighborhood, replace the current label
            # with this most frequent one
            most_freq = C.most_common(1)[0]
            if most_freq[1]>limits+1:
                reduced_labels[i]=most_freq[0]
        except IndexError:
            pass
        
    # only consider the segments where the user is walking
    where_walking = reduced_labels==1
    # extract the walking intervals
    intervals = extract_intervals(where_walking)
    
    # only consider those longer than thresh_min_duration
    # thresh is in seconds, so it is necessary to convert the length
    # in windows of the segments to length in seconds
    # calculate the average legnth of a STFT window:
    n_windows = len(shortFT_times)
    avg_window_length = (shortFT_times[-1]-shortFT_times[0])/n_windows
    
    # only consider intervals longer than the threshold
    long_int = (intervals[:,0]*avg_window_length) > thresh_min_duration
    long_intervals = intervals[long_int]
    
    # grab the raw sensor values to segment them
    acc_sensor = np.copy(sessionData[0])
    gyro_sensor = np.copy(sessionData[1])
    mag_sensor = np.copy(sessionData[2])
    
    # convert timestamp to seconds
    acc_sensor[:,0] /= 1000000000
    gyro_sensor[:,0] /= 1000000000
    mag_sensor[:,0] /= 1000000000
    
    # extract each segment and store it in a list
    segments = []
    new_labels = np.zeros(np.shape(reduced_labels))
    
    #iterate through each interval
    for interval in long_intervals:
        start_index = interval[1]
        length = interval[0]
        end_index = start_index + length
        new_labels[start_index:end_index] = 1
        #get the start and end times of each interval (in sec)
        start_time = start_index * avg_window_length
        end_time = end_index * avg_window_length
        
        #get the sensor readings for said time interval
        acc_seg = acc_sensor[np.logical_and(acc_sensor[:,0]>start_time,
                                            acc_sensor[:,0]<end_time)]
        gyro_seg = gyro_sensor[np.logical_and(gyro_sensor[:,0]>start_time,
                                              gyro_sensor[:,0]<end_time)]
        mag_seg = mag_sensor[np.logical_and(mag_sensor[:,0]>start_time, 
                                            mag_sensor[:,0]<end_time)]
        segments.append((acc_seg,gyro_seg,mag_seg))
        
    # if the user wishes to plot
    if plot:
        time = shortFT_times
        freq = shortFT_freqs
        
        plt.figure()
        # plot the accelerometer readings
        ax1 = plt.subplot(3,1,1)
        plt.plot(sessionData[0][:,0]/1000000000, sessionData[0][:,1:])
        
        plt.subplot(3,1,2, sharex=ax1)
        # plot the STFT of the gyro readings
        extent = np.min(time), np.max(time), np.min(freq), np.max(freq)
        plt.imshow(log_energy, extent=extent, aspect='auto')
        
        plt.subplot(3,1,3, sharex=ax1)
        # plot the original KKMeans labels and the reduced ones
        plt.plot(time, new_labels)
        
    return segments

def extract_intervals(labelVector):
    # returns a matrix:
    # each line represents a sequence of true values (walking)
    # col 0 is the size of the sequence
    # col 1 is the start index of said sequence
    n = len(labelVector)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(labelVector[1:] != labelVector[:-1])     # pairwise unequal (string safe)
        j = np.append(np.where(y), n - 1)   # must include last element posi
        length = np.diff(np.append(-1, j))       # run lengths
        indices = np.cumsum(np.append(0, length))[:-1] # positions
    labs = labelVector[j]
    seq_matrix =  np.array((length, indices, labs)).T
    true_seqs = np.where(seq_matrix[:,2]==1)
    true_matrix = seq_matrix[true_seqs]
    return true_matrix[:,0:2]

def preprocess_walk_segment(walkSeg_original, FS = 100):
    # performs linear interpolation at a fixed sampling rate
    walkSeg = list(walkSeg_original)
    min_timestamp = np.min([np.min(sensor[:,0]) for sensor in walkSeg])
    max_timestamp = np.max([np.max(sensor[:,0]) for sensor in walkSeg])
    diff = max_timestamp - min_timestamp
    
    interpTime = np.arange(0, diff, 1/FS)
    
    interpSensors = []
    for sensor in walkSeg: 
        interpSensor = np.zeros((len(interpTime),4))
        interpSensor[:,0] = interpTime
        for axis in [1,2,3]:
            interpSensor[:,axis] = np.interp(interpTime, sensor[:,0]-min_timestamp, sensor[:,axis])
        interpSensors.append(interpSensor)
    
    return tuple(interpSensors)

def divide_walk_segment(walkSeg, length = 100, stride = 50):
    for sensor in walkSeg:
        windows = [sensor[i*stride:i*stride+length,:] for i in range(np.shape(sensor)[0] // stride)]
    return 0
    
def window_feature_extraction(walkSeg_window):
    # based on Nickel et al.
    
    FEATURES = [] #feature list
    # statistical features:
    
    
    
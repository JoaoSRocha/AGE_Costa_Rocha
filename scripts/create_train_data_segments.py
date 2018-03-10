import os
import pickle
import numpy as np

PICKLED_DATA_PATH = 'pickled_data/'
SEGMENTS_PATH = 'train_data_segments/'

os.chdir('..')

pickled_list = os.listdir(PICKLED_DATA_PATH)

for file in pickled_list:
    print(file)
    with open(PICKLED_DATA_PATH+file, 'rb') as pkl:
        data = pickle.load(pkl)
    shape = np.shape(data)
    length = shape[0]
    nSegments = length//300
    device_segments = []
    for i in range(nSegments):
        disp = 300*i
        segment = data[disp:300+disp,:]
        device_segments.append(segment)
    with open(SEGMENTS_PATH+file, 'wb') as out:
        pickle.dump(device_segments,out)
        print('saved')

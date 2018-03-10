import pickle
with open('test_pickle/100006.pkl', 'rb') as f:
    data = pickle.load(f)    
    
import matplotlib.pyplot as plt

from scipy.fftpack import fft
import numpy as np
plt.plot(data[:,1])
ft = fft(data[:,1])
plt.figure()
plt.plot(np.imag(ft))
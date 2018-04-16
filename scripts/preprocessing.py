import scipy.signal as sci
import numpy as np

def moving_average(signal):
    numerator=[1/9,2/9,3/9,2/9,1/9]
    denominator=1
    filter_out=sci.lfilter(numerator,denominator,signal)
    return filter_out
def average_cycle_length_detection(signal,segment_length):
    cycle_0=signal[round(len(signal))/2:len(signal)+segment_length]


def cycle_eval(set1,set2):
    dist = abs(set1-set2)
    return dist

def define_eval_windows(baseline_signal,number_of_windows,segment_length):
    window_array=[]
    for i in baseline_signal:
        window=[baseline_signal[i]:baseline_signal[i+segment_length]
        i=i+segment_length
        window_array.append(window)
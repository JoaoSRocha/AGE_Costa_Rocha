import scipy.signal as sci

def moving_average(signal):
    numerator=[1/9,2/9,3/9,2/9,1/9]
    denominator=1
    filter_out=sci.lfilter(numerator,denominator,signal)
    return filter_out
def average_cycle_length_detection(signal):
    cycle_0=signal[round(len(signal))/2:len(signal)+70]


def cycle_eval(set1,set2):
    dist = abs(set1-set2)
    return dist

def define_eval_windows(signal,number_of_windows):

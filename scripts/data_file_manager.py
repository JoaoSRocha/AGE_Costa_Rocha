import pickle
import os
import numpy
import pandas as pd


def importCSV(path,rows_to_skip):
    data_frame = pd.read_csv(path,skiprows=rows_to_skip)
    return data_frame


def pickleCSV(data_frame,path,name,func):

    if not os.path.exists(path):
        os.makedirs(path)

    if func.lower() in ["save"]:
        pickle_out = open(path+name,"wb")
        pickle.dump(data_frame,pickle_out)
        pickle_out.close()
        print("saved to"+"  "+path)
    if func.lower() in ["load"]:
        pickle_in= open(path+name,"rb")
        data_frame=pickle.load(pickle_in)
        pickle_in.close()
        print("load done")
        return data_frame
#path1="C:/Users/joao/Desktop/AGE/train.csv"
#path2="D:/Users/joao/AGE_project/venv/pickled_data"
#name="/test.pickle"



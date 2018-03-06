import data_file_manager
import numpy as np

path2="D:/Users/joao/AGE_project/venv/pickled_data"
# name="/test.pickle"
#
# a=np.matrix(data_file_manager.pickleCSV(0,path2,name,"load"))
#
#
# time_xyz_data=a[:,0:4]
#
# device=a[:,4]
path="C:/Users/joao/Desktop/AGE/train.csv"
trainfile=np.array(data_file_manager.importCSV(path,0))

device_id=np.unique(trainfile[:,-1])

for device in device_id:
    condition=(trainfile[:,-1]==device)
    print(device)
    seq_i=np.where(condition)
    seq=trainfile[seq_i]

    #seq=np.extract(trainfile[:,-1]==device,trainfile)
    # print(np.shape(seq))
    # print(seq)
    data_file_manager.pickleCSV(seq,path2,"/"+str(int(device))+".pkl","save")


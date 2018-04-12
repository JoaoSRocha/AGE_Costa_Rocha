import AGEmodule
import numpy as np
import matplotlib.pyplot as plt
dataset_path="D:/Users/joao/PycharmProjects/Trainset/"
PICKLE_PATH = 'D:/Users/joao/PycharmProjects/Pickled_dataset/'
USER_id=3

##AGEmodule.create_pickles(dataset_path)

raw_data=AGEmodule.load_user(USER_id)

# walking_data=AGEmodule.walk_detection(raw_data[0])
#
# print(len(walking_data[0][0][:,1]))
#
AGEmodule.plot_session(raw_data[0])

plt.show()
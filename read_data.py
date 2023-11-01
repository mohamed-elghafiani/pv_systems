import scipy.io as sio
import numpy as np
from utils import process_to_dict, get_correct_key, get_max_num_points, get_min_num_points
import re
# import matplotlib.pyplot as plt


DATA_DIRS = ["data/ombrage", "data/Rp", "data/Rs"]
OMB_DATA = []

data_type_key = "resultsV"
data = {}
for dir_ in DATA_DIRS:
    for i in range(1, 6):
        # load data from data.mat file
        OMB_DATA_1 = sio.loadmat(f"{dir_}/{i}/data{i}.mat")
        # create a key for the data such as: 
        # "OM" --> "ombrage"
        # "RS" --> "Rs" ... etc.
        key = re.findall(r"\w+/(\w{2})", dir_)[0].upper() + f"_{i}"
        # get the right key for data we need to read
        # Problem is not always we have the same key: 
        # resultsI2? results4? results3? ...
        data_key = get_correct_key(OMB_DATA_1.keys(), data_type_key)
        # restructure the data in the form of a dictionary such as:
        # {
        #    "OM": [LIST OF OMBRAGE-TYPE DEFECT SIMULATION DATA],
        #    "RS": [LIST OF RS-TYPE DEFECT SIMULATION DATA]
        # }
        process_to_dict(OMB_DATA_1[data_key][0], key, data)


OMB_DATA_1 = sio.loadmat("data/sans_defaut/1/data.mat")
process_to_dict(OMB_DATA_1[data_type_key][0], "WO", data)


max_num = get_max_num_points(data)
min_num = get_min_num_points(data)
print(max_num)
print(min_num)

print(data.keys())

for key in data.keys():
    max_ = 450
    # max_ = len(max(data[key], key=lambda el: len(el)))
    data_homog = []
    for el in data[key]:
        el = list(el)
        data_homog.append([el[0]] * (max_ - len(el)) + el)
    data[key] = data_homog
    

# A check for if data is homogeneous 
len_0 = len(data_homog[0])
for el in data_homog:
    if len(el) != len_0:
        print("differnt length detected!")
        print(f"===> {len(el)}")

sio.savemat("V_DATA.mat", data)

new_data = sio.loadmat("V_DATA.mat")
print(new_data.keys())

lengths = []
for key in [k for k in new_data.keys() if not k.startswith("__")]:
    lengths.append(len(new_data[key][0]))

print(new_data["OM_1"][0])
print(lengths)



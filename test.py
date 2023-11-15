"""This module is for testing"""

import scipy.io as sio
import numpy as np

data = sio.loadmat("matlab/data.mat")['shuffled_data']
# print(data.keys())
# print(data['data'][0][0][0]/max(data['data'][0][0][0]))
# print(data["OM_1"].shape)

data_ = []
for item in data:
    item_ = [item[0][0]/max(item[0][0]), item[1]/max(item[1][0]), item[2][0]]
    data_.append(item_)

# data_ = np.array(data_, dtype=np.float32)

sio.savemat("data_norm.mat", {"data": data_})

new_data = sio.loadmat("data_norm.mat")
print(new_data["data"].shape)
print(max(new_data['data'][0][0][0]))
print(max(new_data['data'][0][1][0]))

# print(len(data_), len(data_[0]))
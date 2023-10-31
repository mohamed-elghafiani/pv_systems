"""This module is for testing"""

import scipy.io as sio

data = sio.loadmat("data1/sans_defaut/1/data.mat")
print(data["tout"].shape)
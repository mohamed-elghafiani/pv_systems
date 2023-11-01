"""This module is for testing"""

import scipy.io as sio

data = sio.loadmat("V_DATA.mat")
print(data.keys())
print(data["OM_1"].shape)
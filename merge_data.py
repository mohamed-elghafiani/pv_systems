import scipy.io as sio
import numpy as np

dataI = sio.loadmat("an_I_data.mat")
dataV = sio.loadmat("V_DATA.mat")

data = []

for key in dataI.keys():
    if not key.startswith("__"):
        for idx in range(len(dataI[key])):
            i, label = dataI[key][idx]
            v = dataV[key][idx]
            data.append([i[0], v, label[0]])


# data = np.array(data, dtype=np.float32)

# sio.savemat("data.mat", {"data": data})

new_data = sio.loadmat("data.mat")
print(new_data["data"].shape)
print(new_data["data"][0][2])
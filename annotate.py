import scipy.io as sio


data = sio.loadmat('I_DATA.mat')

an = {
    'WO': [1.0, 0.0, 0.0, 0.0], 
    'OM': [0.0, 1.0, 0.0, 0.0], 
    'RS': [0.0, 0.0, 1.0, 0.0], 
    'RP': [0.0, 0.0, 0.0, 1.0]
}

# for key in data.keys():
#     d = data[key]
#     new_d = []
#     if not key.startswith('__'):
#         for el in d:
#             new_d.append([el, an[key[:2]]])

#     data[key] = new_d

# sio.savemat('an_I_data.mat', data)

new_data = sio.loadmat('an_I_data.mat')
print()
print(new_data['RS_1'][0])
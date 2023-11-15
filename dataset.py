from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split


def loader():
    BATCH_SIZE = 16
    
    # data = sio.loadmat("matlab/data.mat")["shuffled_data"]
    data = sio.loadmat("data_norm.mat")["data"]
    train_data, test_data = train_test_split(data, test_size=.2)

    # transform list of lists to torch tensor
    def to_tensor(data_):
        data_I = []
        data_V = []
        data_label = []
        for el in data_:
            data_I.append(list(el[0][0]))
            data_V.append(list(el[1][0]))
            data_label.append(list(el[2][0]))

        data_I = torch.tensor(data_I, dtype=torch.float32)
        data_V = torch.tensor(data_V, dtype=torch.float32)
        data_label = torch.tensor(data_label, dtype=torch.float32)
        tr_data = TensorDataset(data_I, data_V, data_label)
        return tr_data

    train_data = to_tensor(train_data)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data = to_tensor(test_data)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    return test_loader, train_loader

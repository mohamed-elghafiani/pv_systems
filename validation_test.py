import torch
from model import Model
import scipy.io as sio
import os
import sys

model = Model()

data = sio.loadmat('data/val.mat')

model_filenam = os.listdir(os.getcwd() + "/checkpoint0")[2]
model_dir = os.getcwd() + f"/checkpoint0/{model_filenam}"

# Load the pretrained weights
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['state_dict'])

I = [list(el)[0] for el in data['I']]
[I.append(el) for el in [I[0]] * (450 - len(I))]

V = [list(el)[0] for el in data['V']]
[V.append(el) for el in [V[0]] * (450 - len(V))]


I = torch.tensor([I], dtype=torch.float32)
V = torch.tensor([V], dtype=torch.float32)

print(model(I, V))

# print(I)
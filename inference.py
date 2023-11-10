from dataset import loader
import torch
from model import Model
import os


data = loader()[0]

model = Model()

model_filenam = os.listdir(os.getcwd() + "/checkpoint0")[2]
model_dir = os.getcwd() + f"/checkpoint0/{model_filenam}"

# Load the pretrained weights
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['state_dict'])

pred, labels = [], []
for (b_i, b_v, b_y) in data:
    pred.append(model(b_i, b_v))
    labels.append(b_y)

pred = torch.cat(pred, dim=0)
labels = torch.cat(labels, dim=0)

print(pred[1])
print(labels[1])
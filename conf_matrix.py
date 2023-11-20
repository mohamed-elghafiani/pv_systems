from dataset import loader
import torch
from model import Model
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


torch.set_printoptions(precision=5)

data = loader()[0]

model = Model()

model_filenam = os.listdir(os.getcwd() + "/checkpoint0")[2]
model_dir = os.getcwd() + f"/checkpoint0/{model_filenam}"

# Load the pretrained weights
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['state_dict'])

pred, labels = [], []
for (b_i, b_v, b_y) in data:
    print((torch.max(torch.exp(model(b_i, b_v)), 1)[1]))

    pred.append(model(b_i, b_v).detach().numpy())
    labels.append(b_y.detach().numpy())

y_pred = []
y_true = []

# iterate over test data
for (b_i, b_v, b_y) in data:
        output = model(b_i, b_v) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).cpu().numpy()
        y_pred.extend(output) # Save Prediction

        b_y = (torch.max(torch.exp(b_y), 1)[1]).cpu().numpy()
        # b_y = b_y.cpu().numpy()
        y_true.extend(b_y) # Save Truth

# constant for classes
classes = ('OM', 'RS', 'RP', 'SD')

# Build confusion matrix
print(y_true)
print(y_pred)
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                    columns = [i for i in classes])

plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')
# plt.show()

print(pred[1])
print(labels[1])
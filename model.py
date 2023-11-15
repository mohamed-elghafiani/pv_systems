from torch import nn 
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.inputI = nn.Linear(450, 500)
        self.rI1 = nn.ReLU()
        self.layerI1 = nn.Linear(500, 500)
        self.rI2 = nn.ReLU()
        self.layerI2 = nn.Linear(500, 500)
        self.rI3 = nn.ReLU()
        self.layerI3 = nn.Linear(500, 500)
        self.rI4 = nn.ReLU()
        self.outputI = nn.Linear(500, 500)

        self.inputV = nn.Linear(450, 500)
        self.rV1 = nn.ReLU()
        self.layerV1 = nn.Linear(500, 500)
        self.rV2 = nn.ReLU()
        self.layerV2 = nn.Linear(500, 500)
        self.rV3 = nn.ReLU()
        self.layerV3 = nn.Linear(500, 500)
        self.rV4 = nn.ReLU()
        self.outputV = nn.Linear(500, 500)

        self.layer1 = nn.Linear(1000, 1000)
        self.r1 = nn.ReLU()
        self.layer2 = nn.Linear(1000, 1000)
        self.r2 = nn.ReLU()
        self.layer3 = nn.Linear(1000, 1000)
        self.r3 = nn.ReLU()
        self.layer4 = nn.Linear(1000, 1000)
        self.r4 = nn.ReLU()
        self.layer5 = nn.Linear(1000, 1000)
        self.r5 = nn.ReLU()
        self.output = nn.Linear(1000, 4)
        self.soft_max = nn.Softmax()

    def forward(self, x, y):
        i1 = self.inputI(x)
        i1 = self.rI1(i1)
        i2 = self.layerI1(i1)
        i2 = self.rI2(i2)
        i3 = self.layerI2(i2) 
        i3 = self.rI3(i3)
        i4 = self.layerI3(i3) 
        i4 = self.rI4(i4)
        i_out = self.outputI(i4)

        v1 = self.inputV(y)
        v1 = self.rV1(v1)
        v2 = self.layerV3(v1) 
        v2 = self.rV2(v2)
        v3 = self.layerV2(v2) 
        v3 = self.rV3(v3)
        v4 = self.layerV3(v3) 
        v4 = self.rV4(v4)
        v_out = self.outputV(v4)

        iv = torch.cat((i_out, v_out), dim=1)
        iv_h1 = self.layer1(iv)
        iv_h1 = self.r1(iv_h1)
        iv_h2 = self.layer2(iv_h1)
        iv_h2 = self.r2(iv_h2)
        iv_h3 = self.layer3(iv_h2)
        iv_h3 = self.r3(iv_h3)
        iv_h4 = self.layer4(iv_h3)
        iv_h4 = self.r4(iv_h4)
        iv_h5 = self.layer4(iv_h4)
        iv_h5 = self.r4(iv_h5)
        iv_h6 = self.layer4(iv_h5)
        iv_h6 = self.r4(iv_h6)
        out = self.output(iv_h6)
        out = self.soft_max(out)

        return out

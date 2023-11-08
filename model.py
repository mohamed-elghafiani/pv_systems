from torch import nn 


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.input = nn.Linear(450, 500)
        self.layer2 = nn.Linear(500, 500)
        self.layer3 = nn.Linear(500, 500)
        self.layer4 = nn.Linear(500, 500)
        self.output = nn.Linear(500, 4)

    def forward(self, x):
        yh1 = self.input(x) 
        yh2 = self.layer2(yh1) 
        yh3 = self.layer3(yh2) 
        yh4 = self.layer4(yh3) 
        y = self.output(yh4)

        return y

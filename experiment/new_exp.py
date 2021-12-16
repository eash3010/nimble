import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, base=4000, num_layers=10):
        super().__init__()

        #layers = []
        self.num_layers = num_layers
        self.layers = []
        self.conv1 = nn.Conv2d(3, base, (2, 2), padding=(1,1))
        #layers.append(conv1)

        #pool = nn.MaxPool2d((2, 2))
        #dropout = nn.Dropout(.2)

        #for i in range(num_layers):
            #conv1 = nn.Conv2d(base+i, base+i+1, (3,3), padding=(1,1))
            #layers.append(conv1)
            #layers.append(dropout)

        self.fc1 = nn.Linear(10000, base)
        self.fc2 = nn.Linear(10000, base)

        for i in range(self.num_layers):
            fc1 = nn.Linear(base-i, base-i-1)
            self.layers.append(fc1)

        self.layers2 = []
        for i in range(1, self.num_layers):
            fc2 = nn.Linear(base-i, base-i-2)
            self.layers2.append(fc2)

        self.layers = nn.ModuleList(self.layers)
        self.layers2 = nn.ModuleList(self.layers2)

        #self.fc1 = nn.Linear((base+num_layers)*225*225, 64)
        #self.fc2 = nn.Linear(64, 10)

        #self.seq_model = nn.Sequential(*layers)

    def forward(self, x):

        #x = self.seq_model(x)
        #print(x.size())
        #x = torch.flatten(x)
        #print(x.size())
        #x = self.fc1(x)
        #print(x.size())
        #x = self.fc2(x)
        print("x",x.size())
        x = self.conv1(x)
        x = torch.flatten(x)
        x1 = None
        x2 = self.fc2(x)
        x = self.fc1(x)
        for i in range(self.num_layers):
            tmp = x+x1
            x = self.layers[i](tmp)

        return x

class BranchedModel(nn.Module):
    def __init__(self, base=4, num_layers=30):
        super().__init__()

        self.layers = []
        self.layers2 = []
        self.conv1 = nn.Conv2d(3, base, (2, 2), padding=(1,1))
        self.conv2 = nn.Conv2d(3, base, (20, 20), padding=(1,1))
        self.conv3 = nn.Conv2d(3, base, (15, 15), padding=(1,1))

        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(.2)

        for i in range(num_layers):
            conv1 = nn.Sequential(
                nn.Conv2d(base, base*2, (3,3), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2,2)),
                nn.Dropout(.4),

                nn.Conv2d(base*2, int(base*2.5), (4, 5), padding=(2,2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2,2)),
                #nn.Dropout(.2),
            )

            conv2 = nn.Sequential(
                nn.Conv2d(int(base*2.5), int(base*1.5), (4, 5), padding=(2,2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2,2)),
                nn.Dropout(.4),

                nn.Conv2d(int(base*1.5), int(base/2), (2, 2), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2,2))
            )
  
            self.layers.append(conv1)
            self.layers2.append(conv2)
            #layers.append(dropout)

        self.conv4 = nn.Conv2d(int(base*2.5), int(base*2.5), (2, 2), padding=(1,1))
        self.layers = nn.ModuleList(self.layers)
        self.layers2 = nn.ModuleList(self.layers2)

        self.fc1 = nn.Linear(89480, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):

        outputs3 = self.pool(self.conv2(x))
        outputs4 = self.pool(self.conv3(x))
        x = self.conv1(x)
        #print("x", x.size())
        outputs = self.layers[0](x)
        #print(outputs.size())
        #outputs = torch.flatten(outputs)
        #print(outputs.size())
        for layer in self.layers[1:]:
            #print(type(outputs))
            outputs = outputs + layer(x)
        #print("outputs", outputs.size())
        outputs=self.conv4(outputs)
        #print("outputs", outputs.size())
        outputs2 = self.layers2[0](outputs)
        #print("outputs2", outputs2.size())
        outputs2 = torch.flatten(outputs2)
        for layer in self.layers2[1:]:
            outputs2 = torch.cat([outputs2, torch.flatten(layer(outputs))])
        outputs2 = torch.cat([outputs2, torch.flatten(outputs3)])
        outputs2 = torch.cat([outputs2, torch.flatten(outputs4)])
        #print(outputs2.size())
        #x = torch.stack(outputs, dim=0)
        #print(x.size())
        #x = torch.flatten(x)
        #print(x.size())
        x = self.fc1(outputs2)
        #print(x.size())
        x = self.fc2(x)
        x = self.fc3(x)
        #print(x.size())
        return x

class SimpleBranchedModel(nn.Module):
    def __init__(self, base=4, num_layers=30):
        super().__init__()

        self.layers = []
        self.layers2 = []
        self.conv1 = nn.Conv2d(3, base, (2, 2), padding=(1,1))

        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(.2)

        for i in range(num_layers):
            conv1 = nn.Sequential(
                nn.Conv2d(base, base*2, (5,5), padding=(2,2)),
                nn.ReLU(inplace=True),
                #nn.MaxPool2d((2,2)),
                nn.Dropout(.4),

                nn.Conv2d(base*2, base, (5, 7), padding=(2,3)),
                nn.ReLU(inplace=True),
                #nn.MaxPool2d((2,2)),
                #nn.Dropout(.2),
            )

            self.layers.append(conv1)

        self.conv4 = nn.Conv2d(base, int(base*2.5), (2, 2), padding=(1,1))
        self.layers = nn.ModuleList(self.layers)

        self.fc1 = nn.Linear(17640, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):

        x = self.conv1(x)
        #print("x", x.size())
        outputs = self.layers[0](x)
        #print(outputs.size())
        for layer in self.layers[1:]:
            #print(type(outputs))
            outputs = outputs + layer(x)
        #print("outputs", outputs.size())
        outputs=self.conv4(outputs)
        #print("outputs", outputs.size())
        x = torch.flatten(outputs, 1)
        #print("X", x.size())
        x = self.fc1(x)
        #print(x.size())
        x = self.fc2(x)
        x = self.fc3(x)
        #print(x.size())
        return x

class SimpleLinearModel(nn.Module):
    def __init__(self, base=4, num_layers=30):
        super().__init__()

        self.layers = []
        self.layers2 = []
        self.conv1 = nn.Conv2d(3, base, (2, 2), padding=(1,1))

        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(.2)

        for i in range(num_layers):
            conv1 = nn.Sequential(
                nn.Conv2d(base, base*2, (5,5), padding=(2,2)),
                nn.ReLU(inplace=True),
                #nn.MaxPool2d((2,2)),
                nn.Dropout(.4),

                nn.Conv2d(base*2, base, (5, 7), padding=(2,3)),
                nn.ReLU(inplace=True),
                #nn.MaxPool2d((2,2)),
                #nn.Dropout(.2),
            )

            self.layers.append(conv1)

        self.conv4 = nn.Conv2d(base, int(base*2.5), (2, 2), padding=(1,1))
        self.layers = nn.Sequential(*self.layers)

        self.fc1 = nn.Linear(510760, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):

        x = self.conv1(x)
        #print("x", x.size())
        outputs = self.layers(x)
        #print("outputs", outputs.size())
        outputs=self.conv4(outputs)
        #print("outputs", outputs.size())
        x = torch.flatten(outputs)
        #print(x.size())
        x = self.fc1(x)
        #print(x.size())
        x = self.fc2(x)
        x = self.fc3(x)
        #print(x.size())
        return x


class SimpleDoubleModel(nn.Module):
    def __init__(self, base = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(3, base, (2, 2), padding=(1,1))
        self.conv2 = nn.Conv2d(3, base, (2, 2), padding=(1,1))

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        #print("x", x.size())
        return x1, x2


class SimpleSmallBranchedModel(nn.Module):
    def __init__(self, base=1, num_layers=4):
        super().__init__()

        self.layers = []
        self.layers2 = []
        self.conv1 = nn.Conv2d(3, base, (2, 2), padding=(1,1))

        for i in range(num_layers):
            conv1 = nn.Sequential(
                nn.Conv2d(base, base*2, (2,2), padding=(2,2)),
                nn.ReLU(inplace=True),

                nn.Conv2d(base*2, base, (2, 2), padding=(2,3)),
                nn.ReLU(inplace=True),
            )

            self.layers.append(conv1)

        self.conv4 = nn.Conv2d(base, int(base*2.5), (2, 2), padding=(1,1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):

        x = self.conv1(x)
        outputs = self.layers[0](x)
        for layer in self.layers[1:]:
            outputs = outputs + layer(x)
        outputs=self.conv4(outputs)
        return outputs

class PaperModel(nn.Module):
    def __init__(self, base = 4):
        super().__init__()

        self.conv1 = nn.Conv2d(3, base, (3, 3), padding=(1,1))
        self.conv2 = nn.Conv2d(base, base, (3, 3), padding=(1,1))
        self.conv3 = nn.Conv2d(base, base, (3, 3), padding=(1,1))
        self.conv4 = nn.Conv2d(base, base, (3, 3), padding=(1,1))
        self.conv5 = nn.Conv2d(base, base, (3, 3), padding=(1,1))
        self.conv6 = nn.Conv2d(base, base, (3, 3), padding=(1,1))
        self.conv7 = nn.Conv2d(base, base, (3, 3), padding=(1,1))

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        x4 = self.conv4(x1)
        x5 = self.conv5(x2+x3)
        x6 = self.conv6(x1+x3+x4)
        x7 = self.conv7(x3+x4+x5)
        #print("x", x.size())
        return x6, x7





def doublemodel():
    return SimpleDoubleModel()


def paper():
    return PaperModel()


def linear(num=1, base=4):
    return SimpleLinearModel(base, num)

def branched(num = 14, base = 4):
    return SimpleBranchedModel(base, num)

def branchedsimple(num = 15, base = 1):
    return SimpleSmallBranchedModel(base, num)


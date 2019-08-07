import os

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(53824, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model_parameter(self, trainset_info):
        model_name = ("mdl_{}_eph_{}_btcsz_{}.pt").format(trainset_info["dataset_name"],
                                                          trainset_info["epochs"],
                                                          trainset_info["batch_size"])
        model_path = os.path.join(trainset_info["path"], model_name)
        torch.save(self.state_dict(), model_path)
        print("Saved model to " + model_path)
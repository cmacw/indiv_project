import os

import torch
import torch.nn as nn
import torchvision


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, 3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, 3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=3)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 192, 3, stride=3)
        self.conv5_bn = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 256, 3, stride=2)
        self.conv6_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        x = nn.functional.relu(self.conv1_bn(self.conv1(x)))
        print(x.size())
        x = nn.functional.relu(self.conv2_bn(self.conv2(x)))
        print(x.size())
        x = nn.functional.relu(self.conv3_bn(self.conv3(x)))
        print(x.size())
        x = nn.functional.relu(self.conv4_bn(self.conv4(x)))
        print(x.size())
        x = nn.functional.relu(self.conv5_bn(self.conv5(x)))
        print(x.size())
        x = nn.functional.relu(self.conv6_bn(self.conv6(x)))
        print(x.size())

        # print(x.size())
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        print(x.size())
        x = self.fc2(x)
        return x

    def save_model_parameter(self, trainset_info, save_dir):
        model_name = ("mdl_{}_eph_{}_bs_{}.pt").format(trainset_info["dataset_name"],
                                                       trainset_info["epochs"],
                                                       trainset_info["batch_size"])
        model_path = os.path.join(save_dir, model_name)
        torch.save(self.state_dict(), model_path)
        print("Saved model to " + model_path)

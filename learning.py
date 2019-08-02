import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader


class PosEstimationDataset(Dataset):
    def __init__(self, dataset_name, pos_file_name, image_file_name, size, cam_id, transform=None):
        self.dataset_name = dataset_name
        self.pos_file_name = pos_file_name
        self.image_file_name = image_file_name
        self.size = size
        self.cam_id = cam_id
        self.all_pos_euler = self._prepare_pos(pos_file_name, size)
        self.transform = transform

    def __len__(self):
        return self.size

    # Return dictionary of {image, pos}
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_name, self.image_file_name.format(idx, self.cam_id))
        img = plt.imread(img_name)
        pos = self.all_pos_euler[idx, :]

        if self.transform:
            img = self.transform(img)

        sample = {"image": img, "pos": pos}
        return sample

    # read position from csv
    def _read_csv(self, dir_name, size):
        return np.loadtxt(dir_name, delimiter=",")[:size, :]

    # Change rotation matrix [:, 3:12] to euler angles and
    # Return pos and euler angles together as Tensor
    def _prepare_pos(self, pos_file_name, size):
        full_state = self._read_csv(pos_file_name, size)
        rot_mat = np.reshape(full_state[:, 3:], (-1, 3, 3))

        # Transform the rotation to euler
        rot_euler = Rotation.from_dcm(rot_mat).as_euler('zyx', degrees=True)

        # Combine position and euler
        full_state[:, 3:6] = rot_euler / 180
        pos_euler = torch.from_numpy(full_state[:, :6])

        return pos_euler.float()


def show_batch_image(trainloader):
    for i_batch, sample_batched in enumerate(trainloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['pos'].size())

        images_batch = sample_batched["image"]

        if i_batch == 0:
            plt.figure()
            grid = torchvision.utils.make_grid(images_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
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


def plot_array(data):
    plt.figure()
    plt.plot(data)
    plt.ylabel("MSE losses")
    plt.show()


if __name__ == '__main__':
    os.chdir("datasets/Set02/Train")
    dataset_name = "realistic_un"
    cam_id = 0
    ndata = 10000
    epochs = 1
    batch_size = 4

    # Tensor using CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Input data setup
    trsfm = transforms.Compose(
        [transforms.ToTensor()])
    trainset = PosEstimationDataset(dataset_name, "cam_pos.csv",
                                    "image_t_{}_cam_{}.png", ndata, cam_id, transform=trsfm)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Network setup
    net = Net()
    net.to(device)
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Initialise loss array
    losses = np.empty(int(epochs * ndata / batch_size / 100))

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a dictionary of {image, pos}
            image, pos = data['image'].to(device), data['pos'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(image)
            loss = criterion(outputs, pos)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                # save loss
                losses[epoch * ndata + i // 100] = loss

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    plot_array(losses)

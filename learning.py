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
from skimage import io
from torch.utils.data import Dataset, DataLoader


class PosEstimationDataset(Dataset):
    def __init__(self, dataset_name, pos_file_name, image_file_name, size, cam_id, transform=None):
        self.dataset_name = dataset_name
        self.pos_file_name = pos_file_name
        self.image_file_name = image_file_name
        self.size = size
        self.cam_id = cam_id
        self.all_pos = self.read_csv(pos_file_name)
        self.transform = transform

    # function to read load all images
    def read_many_png(self, dir_name, file_name, cam_id, n):
        all_img = []
        for i in range(n):
            all_img.append(np.array(Image.open(
                dir_name + "/" + file_name.format(i, cam_id))))
        return all_img

    # read position from csv
    def read_csv(self, dir_name, n):
        return np.loadtxt(dir_name, delimiter=",")[:n, :]

    def __len__(self):
        return self.size

    # Return dictionary of {image, pos}
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_name, self.image_file_name.format(idx, self.cam_id))
        image = io.imread(img_name)
        # image = self.img2tensor(image)
        pos = self.all_pos[idx, :]
        pos = torch.from_numpy(pos).float()

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "pos": pos}
        return sample

    def _rot_mat2euler(self, mat):
        sy = math.sqrt(mat[0] * mat[0] + mat[3] * mat[3])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(mat[7], mat[8])
            y = math.atan2(-mat[6], sy)
            z = math.atan2(mat[3], mat[0])
        else:
            x = math.atan2(-mat[5], mat[4])
            y = math.atan2(-mat[6], sy)
            z = 0

        return np.array([x, y, z])

    # To transform images in sample to tensors
    def img2tensor(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return image.transpose((2, 0, 1))

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
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    os.chdir("datasets")
    dataset_name = "random_mj"
    cam_id = 0

    ndata = 1000

    trsfm = transforms.Compose(
        [transforms.ToTensor()])

    trainset = PosEstimationDataset(dataset_name, "cam_pos.csv",
                                    "image_t_{}_cam_{}.png", ndata, cam_id, transform=trsfm)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

    # Set up network
    net = Net()
    loss1 = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            image, pos = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data[image])
            loss = loss1(outputs, data[pos])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 99 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
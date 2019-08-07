import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from Net import Net
from PosEstimationDataset import PosEstimationDataset


class PoseEstimation:
    def __init__(self, dataset_info, degrees=True):
        self.dataset_info = dataset_info
        self.degrees = degrees
        # Tensor using CPU or GPU
        self.device = self.use_cuda()

        # Input data setup
        trsfm = transforms.Compose([transforms.ToTensor()])
        self.dataset = PosEstimationDataset(self.dataset_info, transform=trsfm, degrees=degrees)
        self.dataloader = DataLoader(self.dataset, batch_size=self.dataset_info["batch_size"], shuffle=True)

        # model setup
        self.net = Net()
        self.net.to(self.device)
        self.criterion = nn.MSELoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        # Initialise loss array
        self.losses = np.empty(self.dataset_info["epochs"] * len(self.dataloader))

        # Initialise distance and angle diff array
        self.diff = np.empty([self.dataset_info["epochs"] * len(self.dataloader), 2])

    def train(self):
        loss_sample_size = 100

        # Training
        t0 = time.time()
        for epoch in range(self.dataset_info["epochs"]):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.dataloader):

                # get the inputs; data is a dictionary of {image, pos}
                image, pos = data['image'].to(self.device), data['pos'].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(image)
                loss = self.criterion(outputs, pos)
                loss.backward()
                self.optimizer.step()

                # Calculate the difference in euclidean distance and angles
                self.diff[epoch * len(self.dataloader) + i, :] = \
                    np.average(self.cal_diff(outputs, pos, self.degrees), axis=1)
                self.losses[epoch * len(self.dataloader) + i] = loss.item()

                # print statistics
                running_loss += loss.item()
                if i % loss_sample_size == loss_sample_size - 1:  # print every 100 mini-batches
                    # save loss

                    print('[{}, {}] loss: {:.3f}, diff_[distance, angle]: {})'.
                          format(epoch + 1, i + 1, running_loss / loss_sample_size,
                                 self.diff[epoch * len(self.dataloader) + i]))
                    running_loss = 0.0
        t1 = time.time()
        print('\nFinished Training. Time taken: {}'.format(t1 - t0))

    def evaluation(self):
        loss_sample_size = 100
        running_loss = 0

        # turn on evaluation mode
        self.net.eval()

        # start evaluation
        for i, data in enumerate(self.dataloader):

            # get the inputs; data is a dictionary of {image, pos}
            image, pos = data['image'].to(self.device), data['pos'].to(self.device)

            # forward + backward + optimize
            outputs = self.net(image)
            loss = self.criterion(outputs, pos)

            # Calculate the difference in euclidean distance and angles
            self.diff[i, :] = np.average(self.cal_diff(outputs, pos, self.degrees), axis=1)
            self.losses[i] = loss.item()

            # print statistics
            running_loss += loss.item()
            if i % loss_sample_size == loss_sample_size - 1:  # print every 100 mini-batches
                # save loss

                print('[{}] loss: {:.3f}, diff_[distance, angle]: {}'.
                      format(i + 1, running_loss / loss_sample_size,
                             self.diff[i]))
                running_loss = 0.0

        print('\nFinished evalutaion')
        self.print_avg_stat()

    def use_cuda(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("error")
                try:
                    torch.cuda.get_device_capability(device)
                except Exception:
                    device = torch.device("cpu")
        print(device)
        return device

    def show_batch_image(self):
        for i_batch, sample_batched in enumerate(self.trainloader):
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

    # Save model and losses
    def save_model_output(self):
        self.net.save_model_parameter(self.dataset_info)
        self.save_array2csv(self.dataset_info, self.losses, "loss")
        self.save_array2csv(self.dataset_info, self.diff, "diff")

    # Visualise the losses and deviation
    def show_training_fig(self):
        self.plot_array(self.losses, "MSE_losses", self.dataset_info)
        self.plot_array(self.diff[:, 0], "Difference_in_distance(m)", self.dataset_info)
        self.plot_array(self.diff[:, 1], "Difference_in_angle(deg)", self.dataset_info)

    def plot_array(self, data, ylabel, trainset_info):
        plt.figure()
        plt.plot(data)
        plt.ylabel(ylabel)
        fig_name = "fig_{}_eph{}_btcsz{}_{}.png".format(trainset_info["dataset_name"],
                                                          trainset_info["epochs"],
                                                          trainset_info["batch_size"],
                                                          ylabel)
        file_path = os.path.join(trainset_info["path"], fig_name)
        plt.savefig(file_path)
        plt.show()

    def save_array2csv(self, trainset_info, data, name):
        file_name = "{}_{}_eph{}_btcsz{}.csv".format(name,
                                                       trainset_info["dataset_name"],
                                                       trainset_info["epochs"],
                                                       trainset_info["batch_size"])
        file_path = os.path.join(trainset_info["path"], file_name)
        np.savetxt(file_path, data, delimiter=",")

    def cal_diff(self, predict, true, degree=True):
        # predict and ture has size [batch_size, 6]
        # [:, :3] is the translational position
        # [:, 3:] is the rotation in euler angle
        out_np = predict.cpu().detach().numpy()
        pos_np = true.cpu().detach().numpy()

        # Get the euclidean distance
        diff_distances = np.linalg.norm((out_np[:, :3] - pos_np[:, :3]), axis=1)

        # Calculate the rotation angle from predicated(output) to true(input)
        # diff * output = pos
        # diff = pos * inv(output)
        # Since the rotvec is the vector of the axis multplited by the angle
        # The angle is found by finding magnitude of the vector
        out_rot = Rotation.from_euler("zyx", out_np[:, 3:], degrees=degree)
        pos_rot = Rotation.from_euler("zyx", pos_np[:, 3:], degrees=degree)
        rot = pos_rot * out_rot.inv()
        diff_angle = rot.as_rotvec()
        diff_rot = np.linalg.norm(diff_angle, axis=1)
        if degree:
            diff_rot = np.rad2deg(diff_rot)
        return [diff_distances, diff_rot]

    def load_model_parameter(self, path):
        self.net.load_state_dict(torch.load(path))

    def print_avg_stat(self):
        print("avg loss: {:.3f} \t avg[distance, angle] {}".format(np.average(self.losses),
                                                                   np.average(self.diff, axis=0)))
        return np.average(self.losses), np.average(self.diff, axis=0)


if __name__ == '__main__':
    os.chdir("datasets/Set02")

    trainset_info = {"path": "Train", "dataset_name": "realistic_un", "cam_id": 0,
                     "image_name": "image_t_{}_cam_{}.png",
                     "pos_file_name": "cam_pos.csv",
                     "ndata": 15000, "epochs": 25, "batch_size": 4}
    trainer = PoseEstimation(trainset_info, degrees=True)
    # Recover parameters. CHECK BEFORE RUN!!
    # trainer.net.load_state_dict(torch.load("Train/mdl_realistic_un_eph_25_btcsz_4.pt"))
    trainer.train()
    trainer.save_model_output()
    trainer.show_training_fig()


    testset_info = {"path": "Test", "dataset_name": "realistic_un_test", "cam_id": 0,
                    "image_name": "image_t_{}_cam_{}.png",
                    "pos_file_name": "cam_pos_test.csv",
                    "ndata": 1000, "epochs": 1, "batch_size": 1}
    tester = PoseEstimation(testset_info, degrees=True)
    tester.net.load_state_dict(torch.load("Train/mdl_realistic_un_eph_25_btcsz_4.pt"))
    tester.evaluation()

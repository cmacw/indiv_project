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
    def __init__(self, trainset_info, testset_info=None):
        self.trainset_info = trainset_info

        # Tensor using CPU or GPU
        self.device = self.use_cuda()

        # Input data setup
        self.trsfm = transforms.Compose([transforms.ToTensor()])
        self.dataset = PosEstimationDataset(self.trainset_info, transform=self.trsfm)
        self.dataloader = DataLoader(self.dataset, batch_size=self.trainset_info["batch_size"], shuffle=True)

        # Set up testset
        if testset_info is not None:
            self.load_test_set(testset_info)

        # model setup
        self.net = Net()
        self.net.to(self.device)
        self.criterion = nn.MSELoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def load_test_set(self, testset_info):
        self.testset_info = testset_info
        self.testset = PosEstimationDataset(self.testset_info, transform=self.trsfm)
        self.testloader = DataLoader(self.testset, shuffle=True)

    def train(self, show_fig=True, save_output=True, eval_eph=False):
        loss_sample_size = 100

        # Initialise loss array
        losses = np.empty(self.trainset_info["epochs"] * len(self.dataloader))

        # Initialise distance and angle diff array
        eph_diff = np.empty([self.trainset_info["epochs"], 2])

        # Training
        t0 = time.time()
        for epoch in range(self.trainset_info["epochs"]):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.dataloader):
                # Set network to training mode
                self.net.train()

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
                losses[epoch * len(self.dataloader) + i] = loss.item()

                # print statistics
                running_loss += loss.item()
                if i % loss_sample_size == loss_sample_size - 1:  # print every 100 mini-batches
                    # save loss

                    print('[{}, {}] loss: {:.3f}'.
                          format(epoch + 1, i + 1, running_loss / loss_sample_size))
                    running_loss = 0.0

            # Run evaluation and show results
            if eval_eph:
                print('[Epoch', epoch + 1, end='] Test ')
                eph_loss, eph_diff[epoch, :] = self.evaluation()

        t1 = time.time()
        print('\nFinished Training. Time taken: {}'.format(t1 - t0))

        if save_output:
            self.save_model_output(losses, eph_diff)

        if show_fig:
            self.display_training_fig(losses, eph_diff)

    # Evaluation use model in the class to run
    def evaluation(self):
        assert self.testset is not None, \
            "No testset is supplied. Make sure PoseEstimation.load_test_set(set_info) is called"

        loss_sample_size = 100
        running_loss = 0

        # Initialise loss array
        losses = np.empty(len(self.testloader))

        # Initialise distance and angle diff array
        diff = np.empty([len(self.testloader), 2])

        # turn on evaluation mode
        self.net.eval()

        # start evaluation
        for i, data in enumerate(self.testloader):
            # get the inputs; data is a dictionary of {image, pos}
            image, pos = data['image'].to(self.device), data['pos'].to(self.device)

            # forward + backward + optimize
            outputs = self.net(image)
            loss = self.criterion(outputs, pos)

            # Calculate the difference in euclidean distance and angles
            diff[i] = self.cal_diff(outputs, pos)
            losses[i] = loss.item()

        return self.print_avg_stat(losses, diff)

    def use_cuda(self):
        device = torch.device("cpu")
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
    def save_model_output(self, losses, diff):
        self.net.save_model_parameter(self.trainset_info)
        self.save_array2csv(self.trainset_info, losses, "loss")
        self.save_array2csv(self.trainset_info, diff, "diff")

    # Visualise the losses and deviation
    def display_training_fig(self, losses, diff):
        self.plot_array(losses, "MSE_losses", self.trainset_info, scatter=True)
        self.plot_array(diff[:, 0], "Difference_in_distance(m)", self.trainset_info)
        self.plot_array(diff[:, 1], "Difference_in_angle(deg)", self.trainset_info)

    def plot_array(self, data, ylabel, trainset_info, scatter=False):
        plt.figure()
        if scatter:
            x = np.arange(len(data))
            plt.scatter(x, data, s=0.8)
        else:
            plt.plot(data)
        plt.ylabel(ylabel)
        fig_name = "fig_{}_eph{}_btcsz{}_{}.png".format(trainset_info["dataset_name"],
                                                        trainset_info["epochs"],
                                                        trainset_info["batch_size"],
                                                        ylabel)
        file_path = os.path.join(trainset_info["path"], trainset_info["dataset_name"]+"_results", fig_name)
        plt.savefig(file_path)
        plt.show()

    def save_array2csv(self, trainset_info, data, name):
        file_name = "{}_{}_eph{}_btcsz{}.csv".format(name,
                                                     trainset_info["dataset_name"],
                                                     trainset_info["epochs"],
                                                     trainset_info["batch_size"])
        file_path = os.path.join(trainset_info["path"], trainset_info["dataset_name"]+"_results", file_name)
        np.savetxt(file_path, data, delimiter=",")

    def cal_diff(self, predict, true):
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
        out_rot = Rotation.from_euler("zyx", out_np[:, 3:])
        pos_rot = Rotation.from_euler("zyx", pos_np[:, 3:])
        rot = pos_rot * out_rot.inv()
        diff_angle = rot.as_rotvec()
        diff_rot = np.linalg.norm(diff_angle, axis=1)
        diff_rot = np.rad2deg(diff_rot)

        return [diff_distances, diff_rot]

    def load_model_parameter(self, path):
        self.net.load_state_dict(torch.load(path))

    def print_avg_stat(self, losses, diff):
        avg_loss = np.average(losses)
        avg_diff = np.average(diff, axis=0)
        print("avg loss: {:.3f} | avg[distance, angle] {}\n".format(avg_loss, avg_diff))
        return avg_loss, avg_diff


if __name__ == '__main__':
    os.chdir("datasets/Set02")
    datasets = ["random_mj", "random_un", "realistic_mj", "realistic_un"]


    for dataset in datasets:
        trainset_info = {"path": "Train", "dataset_name": dataset, "cam_id": 0,
                         "image_name": "image_t_{}_cam_{}.png",
                         "pos_file_name": "cam_pos.csv",
                         "ndata": 10000, "epochs": 32, "batch_size": 32}

        testset_info = {"path": "Test", "dataset_name": dataset+"_test", "cam_id": 0,
                        "image_name": "image_t_{}_cam_{}.png",
                        "pos_file_name": "cam_pos_test.csv",
                        "ndata": 1000}

        trainer = PoseEstimation(trainset_info)
        # Recover parameters. CHECK BEFORE RUN!!
        # trainer.net.load_state_dict(torch.load("Train/mdl_realistic_un_eph_25_btcsz_4.pt"))
        trainer.load_test_set(testset_info)
        trainer.train(show_fig=True, save_output=True, eval_eph=True)


    # Train one set only
    # trainset_info = {"path": "Train", "dataset_name": "realistic_un", "cam_id": 0,
    #                  "image_name": "image_t_{}_cam_{}.png",
    #                  "pos_file_name": "cam_pos.csv",
    #                  "ndata": 10000, "epochs": 32, "batch_size": 32}
    #
    # testset_info = {"path": "Test", "dataset_name": "realistic_un_test", "cam_id": 0,
    #                 "image_name": "image_t_{}_cam_{}.png",
    #                 "pos_file_name": "cam_pos_test.csv",
    #                 "ndata": 1000}
    #
    # trainer = PoseEstimation(trainset_info)
    # # Recover parameters. CHECK BEFORE RUN!!
    # # trainer.net.load_state_dict(torch.load("Train/mdl_realistic_un_eph_25_btcsz_4.pt"))
    # trainer.load_test_set(testset_info)
    # trainer.train(show_fig=True, save_output=True, eval_eph=True)

    # Make a seperate object for evaluation
    # tester = PoseEstimation(testset_info)
    # tester.net.load_state_dict(torch.load("Train/mdl_realistic_un_eph_25_btcsz_32.pt"))
    # tester.evaluation()

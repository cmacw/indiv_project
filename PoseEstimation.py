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
    def __init__(self, trainset_info, testset_info=None, lr=0.001, wd=0, radial=False):
        self.trainset_info = trainset_info
        self.radial = radial

        # Tensor using CPU or GPU
        self.device = self._use_cuda()

        # model setup
        self.net = Net()
        self.net.to(self.device)
        if radial:
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)

        # Input data setup
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trsfm = transforms.Compose([transforms.Resize((128, 128)),
                                         transforms.ToTensor(),
                                         normalize])
        # self.trsfm = transforms.Compose([transforms.ToTensor()])
        self.trainset = PosEstimationDataset(self.trainset_info, transform=self.trsfm, radial=radial)
        self.norm_range = self.trainset.get_norm_range()
        self.trainloader = DataLoader(self.trainset, batch_size=self.trainset_info["batch_size"], shuffle=True)

        # Set up testset
        if testset_info is not None:
            self.load_test_set(testset_info, radial=radial)

        # initialise directory for saving training results
        self.save_dir = os.path.join(trainset_info["path"],
                                     trainset_info["dataset_name"] + "_results",
                                     "eph{}_bs{}_lr{}_wd{}".format(trainset_info["epochs"],
                                                                   trainset_info["batch_size"],
                                                                   lr, wd))

    def load_test_set(self, testset_info, radial=False, webcam_test=False):
        self.testset_info = testset_info
        self.testset = PosEstimationDataset(self.testset_info, self.trsfm, self.norm_range, radial, webcam_test)
        self.testloader = DataLoader(self.testset, shuffle=True)

    def train(self, show_fig=True, save_output=True, eval_eph=False):
        # Create directory for saving results
        os.makedirs(self.save_dir, exist_ok=False)

        loss_sample_size = len(self.trainloader) // 4

        # Initialise loss array
        train_losses = np.zeros(self.trainset_info["epochs"] * len(self.trainloader))

        # Initialise distance and angle diff array
        eph_losses = np.zeros([self.trainset_info["epochs"], 2])
        eph_diff = np.zeros([self.trainset_info["epochs"], 4])

        # Begin training
        t0 = time.time()
        try:
            for epoch in range(self.trainset_info["epochs"]):  # loop over the dataset multiple times
                print('\n[Epoch', epoch + 1, ']')
                running_loss = 0.0
                for i, data in enumerate(self.trainloader):
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
                    train_losses[epoch * len(self.trainloader) + i] = loss.item()

                    # print statistics
                    # running_loss += loss.item()
                    # if i % loss_sample_size == loss_sample_size - 1:
                    #     print('[{}, {}] loss: {:.5f}'.
                    #           format(epoch + 1, i + 1, running_loss / loss_sample_size))
                    #     running_loss = 0.0

                # Run evaluation and show results
                if eval_eph:
                    eph_losses[epoch], eph_diff[epoch, :] = self.evaluation()

        except KeyboardInterrupt:
            pass

        t1 = time.time()
        print('Time taken: {}'.format(t1 - t0))

        # Save output
        if save_output:
            self.save_model_output(train_losses, eph_losses, eph_diff)

        if show_fig:
            self.display_training_fig(train_losses, eph_losses, eph_diff)

        print('\n--- Finished Training ---\n')

        # Evaluation use model in the class to run

    def evaluation(self):
        assert self.testset is not None, \
            "No testset is supplied. Make sure PoseEstimation.load_test_set(set_info) is called"
        # Initialise loss array
        losses = np.zeros(len(self.testloader))

        # Initialise distance and angle diff array
        diff = np.zeros([len(self.testloader), 2])

        # turn on evaluation mode
        self.net.eval()

        # start evaluation
        for i, data in enumerate(self.testloader):
            # get the inputs; data is a dictionary of {image, pos}
            image, pos = data['image'].to(self.device), data['pos'].to(self.device)

            # forward
            outputs = self.net(image)
            loss = self.criterion(outputs, pos)

            # Calculate the error
            losses[i] = loss.item()
            diff[i] = self.cal_error(outputs, pos)

        print("true   : {}".format(pos[-1]))
        print("predict: {}".format(outputs[-1]))
        return self.print_avg_stat(losses, diff)

    def _use_cuda(self):
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
    def save_model_output(self, train_losses, test_losses, test_diff):
        self.net.save_model_parameter(self.trainset_info, self.save_dir)
        self.save_array2csv(self.trainset_info, train_losses, "train_loss")
        self.save_array2csv(self.trainset_info, test_losses, "eph_loss")
        self.save_array2csv(self.trainset_info, test_diff, "diff")

    # Visualise the losses and deviation
    def display_training_fig(self, train_losses, test_losses, test_diff):
        self.plot_array(train_losses, "Loss", self.trainset_info, scatter=True)
        if self.radial:
            self.plot_array(test_diff[:, 0], "Difference_in_distance(m)", self.trainset_info, std=test_diff[:, 1])
        else:
            self.plot_array(test_diff[:, 0], "Difference_in_distance(m)", self.trainset_info, std=test_diff[:, 2])
            self.plot_array(test_diff[:, 1], "Difference_in_angle(deg)", self.trainset_info, std=test_diff[:, 3])

        avg_train_losses = np.average(train_losses.reshape(-1, len(self.trainloader)), axis=1)
        plt.figure()
        plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label="train")
        plt.plot(range(1, len(test_losses) + 1), test_losses[:, 1], label="test")
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        plt.legend()
        fig_name = "fig_{}_eph{}_bs{}_{}.png".format(self.trainset_info["dataset_name"],
                                                     self.trainset_info["epochs"],
                                                     self.trainset_info["batch_size"],
                                                     "Loss_comp")
        file_path = os.path.join(self.save_dir, fig_name)
        plt.savefig(file_path)

    def plot_array(self, data, ylabel, trainset_info, scatter=False, std=None):
        plt.figure()
        if scatter:
            x = np.arange(len(data))
            plt.plot(x, data, marker='o', markersize=0.6, linewidth='0')
            plt.yscale("log")
            plt.xlabel("batch")
        else:
            plt.errorbar(range(1, len(data) + 1), data, yerr=std, ecolor="k", capsize=3)
            plt.xlabel("epoch")

        plt.ylabel(ylabel)
        fig_name = "fig_{}_eph{}_bs{}_{}.png".format(trainset_info["dataset_name"],
                                                     trainset_info["epochs"],
                                                     trainset_info["batch_size"],
                                                     ylabel)
        file_path = os.path.join(self.save_dir, fig_name)
        plt.savefig(file_path)
        plt.close('all')

    def save_array2csv(self, trainset_info, data, name):
        file_name = "{}_{}_eph{}_bs{}.csv".format(name,
                                                  trainset_info["dataset_name"],
                                                  trainset_info["epochs"],
                                                  trainset_info["batch_size"])
        file_path = os.path.join(self.save_dir, file_name)
        np.savetxt(file_path, data, delimiter=",")

    def cal_error(self, predict, true):

        # predict and ture has size [batch_size, 6]
        # [:, :3] is the translational position
        # [:, 3:] is the rotation in euler angle
        # De-normalise
        predict_np = self._denormalise(predict.cpu().detach().numpy())
        true_np = self._denormalise(true.cpu().detach().numpy())

        if self.radial:
            return predict_np - true_np
        else:
            # Get the euclidean distance
            error_distances = np.linalg.norm((predict_np[:, :3] - true_np[:, :3]), axis=1)

            # Calculate the rotation angle from predicated(output) to true(input)
            # diff * output = pos
            # diff = pos * inv(output)
            # Since the rotvec is the vector of the axis multplited by the angle
            # The angle is found by finding magnitude of the vector
            predict_rot = Rotation.from_quat(predict_np[:, 3:])
            true_rot = Rotation.from_quat(true_np[:, 3:])
            rot = true_rot * predict_rot.inv()
            diff_angle = rot.as_rotvec()
            error_rot = np.linalg.norm(diff_angle, axis=1)
            error_rot = np.rad2deg(error_rot)

            return [error_distances, error_rot]

    def _denormalise(self, pos):
        return pos * (self.norm_range["max"] - self.norm_range["min"]) + self.norm_range["min"]

    def load_model_parameter(self, path):
        self.net.load_state_dict(torch.load(path))

    def print_avg_stat(self, losses, diff):
        avg_loss = np.average(losses)
        avg_diff = np.average(diff, axis=0)
        std_loss = np.std(losses)
        std_diff = np.std(diff, axis=0)
        print(self.trainset.dataset_name)
        print("Test avg loss: {:.5f} | avg[distance, angle] {}".format(avg_loss, avg_diff))
        print("Test std loss: {:.5f} | std[distance, angle] {}".format(std_loss, std_diff))

        return [avg_loss, std_loss], np.concatenate((avg_diff, std_diff), axis=0)

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from PIL import Image
from torch.utils.data import Dataset


class PosEstimationDataset(Dataset):
    def __init__(self, set_info, transform=None, norm_range=None, radial=False, webcam_test=False):
        self.path = set_info["path"]
        self.dataset_name = set_info["dataset_name"]
        self.pos_file_name = set_info["pos_file_name"]
        self.image_file_name = set_info["image_name"]
        self.size = set_info["ndata"]
        self.cam_id = set_info["cam_id"]
        self.transform = transform
        self.norm_range = norm_range
        self.webcam_test = webcam_test
        self.all_pos_euler = self._prepare_pos(self.path + "/" + self.pos_file_name, self.size, radial)

    def __len__(self):
        return self.size

    # Return dictionary of {image, pos}
    def __getitem__(self, idx):
        if self.webcam_test:
            id = int(self.all_pos_euler.cpu().detach().numpy()[idx, 0])
            img_name = os.path.join(self.path, self.dataset_name, self.image_file_name.format(id))
            img = Image.open(img_name)
            pos = self.all_pos_euler[idx, 1].reshape((1))
        else:
            img_name = os.path.join(self.path, self.dataset_name, self.image_file_name.format(idx, self.cam_id))
            # img = plt.imread(img_name)
            # img.show()
            img = Image.open(img_name)

            # Get the 6D pose
            pos = self.all_pos_euler[idx]

            # Show image after transform
            # plt.imshow(img.permute(1, 2, 0))

        if self.transform:
            img = self.transform(img)
        sample = {"image": img, "pos": pos}
        return sample

    # read position from csv
    def _read_csv(self, dir_name, size):
        return np.loadtxt(dir_name, delimiter=",")[:size, :]

    # Change rotation matrix [:, 3:12] to euler angles and
    # Return pos and euler angles together as Tensor
    def _prepare_pos(self, pos_file_path, size, radial=False):
        full_state = self._read_csv(pos_file_path, size)

        if radial:
            # Get the radial distance between the camera and the corner of red cube
            output = np.linalg.norm(full_state[:, :3] - np.array([0.01975, -0.0135, 0.029]), axis=1).reshape((-1,1))
        else:
            # Transform the rotation to euler
            rot_mat = np.reshape(full_state[:, 3:], (-1, 3, 3))
            ort = Rotation.from_dcm(rot_mat).as_quat()

            # Reconstruct the pos and euler array
            output = np.concatenate((full_state[:, :3], ort), axis=1)

        # Get the max and min for angles and position if nor provided
        if self.norm_range is None:
            self.norm_range = {}
            self.norm_range["max"] = output.max(axis=0)
            self.norm_range["min"] = output.min(axis=0)

        # Normalised to [0, 1]
        output_norm = (output - self.norm_range["min"]) / (self.norm_range["max"] - self.norm_range["min"])

        if self.webcam_test:
            output = full_state[:, 1]
            output_norm = (output - self.norm_range["min"]) / (self.norm_range["max"] - self.norm_range["min"])
            output_norm = np.concatenate((full_state[:, 0].reshape((-1,1)), output_norm.reshape((-1,1))), axis=1)

        # Convert to a pytorch tensor
        output_tensor = torch.from_numpy(output_norm)

        return output_tensor.float()

    def get_norm_range(self):
        return self.norm_range


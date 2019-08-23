import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from PIL import Image
from torch.utils.data import Dataset


class PosEstimationDataset(Dataset):
    def __init__(self, set_info, transform=None, pos_range=None, ang_range=None):
        self.path = set_info["path"]
        self.dataset_name = set_info["dataset_name"]
        self.pos_file_name = set_info["pos_file_name"]
        self.image_file_name = set_info["image_name"]
        self.size = set_info["ndata"]
        self.cam_id = set_info["cam_id"]
        self.transform = transform
        self.pos_range = pos_range
        self.ang_range = ang_range
        self.all_pos_euler = self._prepare_pos(self.path + "/" + self.pos_file_name, self.size)

    def __len__(self):
        return self.size

    # Return dictionary of {image, pos}
    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.dataset_name, self.image_file_name.format(idx, self.cam_id))
        # img = plt.imread(img_name)
        img = Image.open(img_name)
        # img.show()
        pos = self.all_pos_euler[idx, :]

        if self.transform:
            img = self.transform(img)

        # Show image after transform
        # plt.imshow(img.permute(1, 2, 0))

        sample = {"image": img, "pos": pos}
        return sample

    # read position from csv
    def _read_csv(self, dir_name, size):
        return np.loadtxt(dir_name, delimiter=",")[:size, :]

    # Change rotation matrix [:, 3:12] to euler angles and
    # Return pos and euler angles together as Tensor
    def _prepare_pos(self, pos_file_path, size):
        full_state = self._read_csv(pos_file_path, size)

        # Transform the rotation to euler
        rot_mat = np.reshape(full_state[:, 3:], (-1, 3, 3))
        rot_euler = Rotation.from_dcm(rot_mat).as_euler('zyx')

        # Get the max and min for angles and position if nor provided
        if self.pos_range is None and self.ang_range is None:
            pos_max, pos_min = full_state[:, :3].max(), full_state[:, :3].min()
            ang_max, ang_min = rot_euler.max(), rot_euler.min()
            self.pos_range = [pos_min, pos_max]
            self.ang_range = [ang_min, ang_max]
        else:
            [pos_min, pos_max] = self.pos_range
            [ang_min, ang_max] = self.ang_range

        # Normalised to [0, 1]. Combine position and euler.
        full_state[:, :3] = (full_state[:, :3] - pos_min) / (pos_max - pos_min)
        full_state[:, 3:6] = (rot_euler - ang_min) / (ang_max - ang_min)

        # Convert to a pytorch tensor
        pos_euler = torch.from_numpy(full_state[:, :6])

        return pos_euler.float()

    def get_angle_range(self):
        return self.ang_range

    def get_pos_range(self):
        return self.pos_range

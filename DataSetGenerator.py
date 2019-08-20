import os
from abc import abstractmethod

import numpy as np
import imageio


class DataSetGenerator:
    def __init__(self, data_set_name, cam_pos_file=None, cam_norm_pos_file=None):
        self.cam_pos = None
        self.cam_norm_pos_file = cam_norm_pos_file
        self.data_set_name = data_set_name
        self.cam_pos_file = cam_pos_file

    @abstractmethod
    def create_data_set(self):
        pass

    def _get_cam_pos(self, radius_range, deg_range, quat, n=100000, start=0):
        # First, either find or create the normalised array
        # Second, scale the normalised array
        # RETURN: n-by-12 np array, 3 for position, 9 for camera orientation in cam_xmat
        r_min, r_max = radius_range
        d_min, d_max = np.asarray(deg_range) / 180 * np.pi

        if self.cam_pos_file:
            pos = np.loadtxt(self.cam_pos_file, delimiter=",")
            assert start < len(pos) and start + n <= len(pos), \
                "The start and/or end index of the cam pos exceed the ones in the position file."
            pos = pos[start:start + n, :]
        else:
            if self.cam_norm_pos_file:
                norm = np.loadtxt(self.cam_norm_pos_file, delimiter=",")[start:start + n, :]
            else:
                norm = np.random.rand(n, 12)
                filename = "cam_norm_pos.csv"
                np.savetxt(filename, norm, delimiter=",")

            # Scale each parameter
            # Col 0: radius, Col 1: angle in horizontal plane,
            # Col 2: angle in vertical plane measure from vertical axis, changes around 45 degrees
            # e.g. 90 degree lies on the horizontal plane
            norm[:, 0] = norm[:, 0] * (r_max - r_min) + r_min
            norm[:, 1] = norm[:, 1] * (d_max - d_min) + d_min
            norm[:, 2] = norm[:, 2] * (d_max - d_min) + d_min + np.pi / 4

            # Translate to xyz and orientation
            pos = np.zeros([n, 12])
            pos[:, 0] = norm[:, 0] * np.cos(norm[:, 1]) * np.sin(norm[:, 2])
            pos[:, 1] = norm[:, 0] * np.sin(norm[:, 1]) * np.sin(norm[:, 2])
            pos[:, 2] = norm[:, 0] * np.cos(norm[:, 2])
            # offset that is added to the rotational matrix
            pos[:, 3:] = (norm[:, 3:] - 0.5) * quat

        self.cam_pos = pos
        return pos

    def _get_debug_cam_pos(self, radius_range, deg_range, quat, n, set_id):
        # 1: Changing texture
        # 2: + radial movement
        # 3: + lateral angle
        # 4: + elevation angle
        # 5: + camera wiggle
        pos = np.zeros([n, 12])
        if set_id == 1:
            coord = np.array([0, -0.354, 0.354])
            pos[:, :3] = np.tile(coord, (n, 1))
        elif set_id == 2:
            range = np.arange(radius_range[0], radius_range[1], 0.05)
            x = np.random.choice(range, n) * np.cos(45 / 180 * np.pi)
            pos[:, 1], pos[:, 2] = -x, x
        elif set_id == 3 or set_id == 4 or set_id == 5:
            r_range = np.arange(radius_range[0], radius_range[1], 0.05)
            d_range = np.arange(-25, 30, 5) / 180 * np.pi
            r = np.random.choice(r_range, n)
            d_horz = np.random.choice(d_range, n)
            d_vert = 45 / 180 * np.pi

            if set_id == 4 or set_id == 5:
                d_vert = np.random.choice(np.arange(25, 70, 5) / 180 * np.pi, n)

            if set_id == 5:
                pos[:, 3:] = (np.random.rand(n, 9) - 0.5) * quat

            pos[:, 0] = r * np.cos(d_horz) * np.sin(d_vert)
            pos[:, 1] = r * np.sin(d_horz) * np.sin(d_vert)
            pos[:, 2] = r * np.cos(d_vert)

        return pos

    def _save_cam_pos(self, pos):
        if self.cam_pos_file is None:
            filename = self.data_set_name + "_cam_pos.csv"
            np.savetxt(filename, pos, delimiter=",")

    def _make_dir(self):
        try:
            os.mkdir(self.data_set_name)
            print("Directory " + self.data_set_name + " created")
        except FileExistsError:
            print("Directory " + self.data_set_name + " already created")

        print("Using " + self.data_set_name + " to store the dataset")

    def _save_fig_to_dir(self, rgb, index, cam_index):
        filename = "image_t_{}_cam_{}.png".format(index, cam_index)
        imageio.imwrite(self.data_set_name + '/' + filename, rgb)

    def print_progress(self, ndata, t):
        # Print progress
        if t % 100 == 0:
            print("Progress: {} / {}".format(t, ndata))

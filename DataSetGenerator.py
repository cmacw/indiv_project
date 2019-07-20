import os
from abc import abstractmethod
from random import uniform

import numpy as np
import scipy.misc


class DataSetGenerator:
    def __init__(self, data_set_name, cam_pos_file=None, cam_norm_pos_file=None):
        self.cam_pos = None
        self.cam_norm_pos_file = cam_norm_pos_file
        self.data_set_name = data_set_name
        self.cam_pos_file = cam_pos_file

    @abstractmethod
    def create_dataset(self, ndata, radius_range, deg_range, quat, cameras, start):
        pass

    def _get_cam_pos(self, radius_range, deg_range, quat, n=100000):
        # First, either find or create the normalised array
        # Second, scale the normalised array
        # RETURN: n-by-12 np array, 3 for position, 9 for camera orientation in cam_xmat
        r_min, r_max = radius_range
        rad_min, rad_max = np.asarray(deg_range) * np.pi / 180
        # rad_min, rad_max = rad_range

        if self.cam_pos_file:
            pos = np.loadtxt(self.cam_pos_file, delimiter=",")[:n, :]
        else:
            if self.cam_norm_pos_file:
                norm = np.loadtxt(self.cam_norm_pos_file, delimiter=",")[:n, :]
            else:
                norm = np.random.rand(n, 12)
                filename = "cam_norm_pos.csv"
                np.savetxt(filename, norm, delimiter=",")

            # Scale each parameter
            # 0: radius, 1: angle in horizontal plane,
            # 2: angle in vertical plane measure from vertical axis
            # e.g. 90 degree points to horizontal plane
            norm[:, 0] = norm[:, 0] * (r_max - r_min) + r_min
            norm[:, 1] = norm[:, 1] * 2 * np.pi
            norm[:, 2] = norm[:, 2] * (rad_max - rad_min) + rad_min

            # Translate to xyz and orientation
            pos = np.zeros([n, 12])
            pos[:, 0] = norm[:, 0] * np.cos(norm[:, 1]) * np.sin(norm[:, 2])
            pos[:, 1] = norm[:, 0] * np.sin(norm[:, 1]) * np.sin(norm[:, 2])
            pos[:, 2] = norm[:, 0] * np.cos(norm[:, 2])
            pos[:, 3:] = (norm[:, 3:] - 0.5) * quat

            filename = "cam_pos.csv"
            np.savetxt(filename, pos, delimiter=",")

        self.cam_pos = pos
        return pos

    def _set_cam_pos(self, cam_name, t, printPos=None):
        # If no
        if self.cam_pos is None:
            self.cam_pos = self._get_cam_pos([0.25, 0.7], [0, 80], 0.5)

        # set position of the reference camera
        cam_id = self.cam_modder.get_camid(cam_name)
        self.model.cam_pos[cam_id] = self.cam_pos[t, 0:3]

        if printPos:
            print("The cam pos is: ", self.cam_pos[t, :])

    # Call after sim.step if want to change the camera orientation while keep
    # pointing to an object
    def _set_cam_orientation(self, cam_name, t, printPos=None):
        cam_id = self.cam_modder.get_camid(cam_name)
        self.sim.data.cam_xmat[cam_id] = self.sim.data.cam_xmat[cam_id] + self.cam_pos[t, 3:]

        if printPos:
            print("The cam orientation is: ", self.cam_pos[t, :])

    def _randomise_light_pos(self):
        x = uniform(-5, 5)
        y = uniform(-5, 5)

        # set position
        # body_pos is hard coded for now
        self.model.light_pos[0, 0] = uniform(-10, 10)
        self.model.light_pos[0, 1] = uniform(-10, 5)

    def _make_dir(self):
        try:
            os.mkdir(self.data_set_name)
            print("Directory " + self.data_set_name + " created")
        except FileExistsError:
            print("Directory " + self.data_set_name + " already created")

        print("Using " + self.data_set_name + " to store the dataset")

    def _save_fig_to_dir(self, rgb, index, cam_index):
        filename = "image_t_{}_cam_{}.png".format(index, cam_index)
        scipy.misc.imsave(self.data_set_name + '/' + filename, rgb)
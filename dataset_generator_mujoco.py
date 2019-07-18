import mujoco_py
from matplotlib import pyplot
from mujoco_py.modder import TextureModder, CameraModder
import math
import os
import scipy.misc
from random import uniform
import numpy as np
import time
from pathlib import Path


class Simulator:
    IMG_SIZE = 512

    def __init__(self, model_path, dataset_name, rand=False, cam_pos_file=None, cam_norm_pos_file=None):
        self.dataset_name = dataset_name
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None)
        self.cam_pos_file = cam_pos_file
        self.cam_norm_pos_file = cam_norm_pos_file
        self.cam_pos = None
        self.tex_modder = TextureModder(self.sim) if rand else None
        self.cam_modder = CameraModder(self.sim) if rand else None

    def on_screen_render(self, cam_name):
        self.sim.reset()
        temp_viewer = mujoco_py.MjViewer(self.sim)
        t = 0

        while True:
            # Randomised material/texture if required
            if self.tex_modder is not None:
                for name in self.sim.model.geom_names:
                    self.tex_modder.rand_all(name)

            # Set camera position and orientation
            self._set_cam_pos(cam_name, t)
            self.sim.step()
            self._set_cam_orientation(cam_name, t)

            temp_viewer.render()
            t += 1
            if t > 100 and os.getenv('TESTING') is not None:
                break

    def create_dataset(self, ndata, r_max, r_min, quat, cameras, start=0):
        self.sim.reset()
        self._make_dir()

        t = start

        # initialise the camera position array
        self.cam_pos = self._get_cam_pos(r_max, r_min, quat, ndata)

        # generate dataset
        while True:

            # Randomised the position of the object
            # Set camera position
            self._set_cam_pos(cameras[0], t)

            # Randomised light source position
            self._randomise_light_pos()

            # Randomised material/texture if required
            if self.tex_modder is not None:
                for name in self.sim.model.geom_names:
                    self.tex_modder.rand_all(name)

            # Simulate and render in offscreen renderer
            self.sim.step()

            # Save images for all camera
            for cam in cameras:
                self._set_cam_orientation(cam, t, True)
                cam_id = self.cam_modder.get_camid(cam)
                self.viewer.render(self.IMG_SIZE, self.IMG_SIZE, cam_id)
                rgb = self.viewer.read_pixels(self.IMG_SIZE, self.IMG_SIZE)[0][::-1, :, :]
                self._save_fig_to_dir(rgb, t, cam_id)

            t += 1
            # Print progress
            if t % 100 == 0:
                print("Progress: {} / {}".format(t, ndata))

            if t == ndata or os.getenv('TESTING') is not None:
                print("Finish creating {} {} images".format(ndata, self.dataset_name))
                break

    def _get_cam_pos(self, r_max, r_min, quat, n=100000):
        # First, either find or create the normalised array
        # Second, scale the normalised array
        # RETURN: n-by-12 np array, 3 for position, 9 for camera orientation in cam_xmat

        if self.cam_pos_file:
            pos = np.loadtxt(self.cam_pos_file, delimiter=",")[:n, :]
        else:
            if self.cam_norm_pos_file:
                norm = np.loadtxt(self.cam_norm_pos_file, delimiter=",")[:n, :]
            else:
                norm = np.random.rand(n, 12)
                filename = "cam_norm_pos.csv"
                np.savetxt(filename, norm, delimiter=",")

            norm[:, 0] = norm[:, 0] * (r_max - r_min) + r_min
            pos = np.zeros([n, 12])
            pos[:, 0] = norm[:, 0] * np.cos(norm[:, 1] * 2 * np.pi) * np.sin(norm[:, 2] * np.pi / 2)
            pos[:, 1] = norm[:, 0] * np.sin(norm[:, 1] * 2 * np.pi) * np.sin(norm[:, 2] * np.pi / 2)
            pos[:, 2] = norm[:, 0] * np.cos(norm[:, 2] * np.pi / 2.1)
            pos[:, 3:] = (norm[:, 3:] - 0.5) * quat

            filename = "cam_pos.csv"
            np.savetxt(filename, pos, delimiter=",")

        self.cam_pos = pos
        return pos

    def _set_cam_pos(self, cam_name, t, printPos=None):
        # If no
        if self.cam_pos is None:
            self.cam_pos = self._get_cam_pos(1, 0.8, 0.01)

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

    # Another trial function to change camera orientation. NOT WORKING
    # def _set_cam_orientation(self, cam, t, printPos=None):
    #     quat = self.cam_modder.get_quat(cam)
    #     self.cam_modder.set_quat(cam, quat + self.cam_pos[t, 3:7])
    #
    #     # Out put cam orientation if needed
    #     if printPos:
    #         print("The cam orientation is: ", self.cam_pos[t, :])

    def _randomise_light_pos(self):
        x = uniform(-5, 5)
        y = uniform(-5, 5)

        # set position
        # body_pos is hard coded for now
        self.model.light_pos[0, 0] = uniform(-10, 10)
        self.model.light_pos[0, 1] = uniform(-10, 5)

    def _make_dir(self):
        try:
            os.mkdir(self.dataset_name)
            print("Directory " + self.dataset_name + " created")
        except FileExistsError:
            print("Directory " + self.dataset_name + " already created")

        print("Using " + self.dataset_name + " to store the dataset")

    def _save_fig_to_dir(self, rgb, index, cam_index):
        filename = "image_t_{}_cam_{}.png".format(index, cam_index)
        scipy.misc.imsave(self.dataset_name + '/' + filename, rgb)


if __name__ == '__main__':
    os.chdir("datasets")
    sim = Simulator("../xmls/box.xml", "text_checker", cam_norm_pos_file="cam_norm_pos.csv", rand=True)

    # preview model
    # sim.on_screen_render("targetcam")

    t0 = time.time()

    # create dataset
    cameras = ["targetcam"]
    sim.create_dataset(15, 0.5, 0.2, 0.05, cameras)

    t1 = time.time()

    print(f"Time to complete: {t1 - t0} seconds")

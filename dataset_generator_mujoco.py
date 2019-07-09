import mujoco_py
from matplotlib import pyplot
from mujoco_py.modder import TextureModder
import math
import os
import scipy.misc
from random import uniform
import numpy as np
import time
from pathlib import Path


class Simulator():

    def __init__(self, model_path, dataset_name, offscreen=False, rand=False):
        self.dataset_name = dataset_name
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.obj_pos = None

        if offscreen:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None)
        else:
            self.viewer = mujoco_py.MjViewer(self.sim)

        if rand:
            self.modder = TextureModder(self.sim)
        else:
            self.modder = None

    def render(self):
        self.sim.reset()
        t = 0
        while True:
            self.sim.step()
            self.viewer.render()
            if t > 100 and os.getenv('TESTING') is not None:
                break

    def create_dataset(self, steps, cameras):
        self.sim.reset()
        self._make_dir()

        t = 0

        # initialise the position array
        self.obj_pos = self._get_obj_pos(filename="testset_mj/testset_mj_obj_norm_pos.csv"
                                         , pos_range=[-5, 5], n=steps)

        # generate dataset
        while True:

            # Randomised the position of the object
            # It is currently hard coded for the x and y position for the cube
            self._set_obj_pos(1, t)

            # Randomised light source position
            self._randomise_light_pos()

            # Randomised material/texture if required
            if self.modder is not None:
                for name in self.sim.model.geom_names:
                    self.modder.rand_all(name)

            # Simulate and render in offscreen renderer
            self.sim.step()

            # Save images for all camera
            for cam in cameras:
                self.viewer.render(1920, 1080, cam)
                rgb = self.viewer.read_pixels(1920, 1080)[0][::-1, :, :]
                self._save_fig_to_dir(rgb, t, cam)

            t += 1
            if t == steps or os.getenv('TESTING') is not None:
                break

    def _get_camera_pos(self, radius, n=100000):
        pos = np.random.rand(n, 3)


    def _get_obj_pos(self, filename=None, pos_range=[-1, 1], n=100000):
        # First, either find or create the normalised array
        # Second, scale the normalised array
        # RETURN: n-by-2 np array
        if filename is None:
            pos = np.random.rand(n, 2)
            filename = self.dataset_name + "/" + self.dataset_name + "_obj_norm_pos.csv"
            np.savetxt(filename, pos, delimiter=",")
        else:
            pos = np.loadtxt(filename, delimiter=",")[:n, :]

        # Scale the normalised array
        pos = pos * (pos_range[1] - pos_range[0]) + pos_range[0]

        # Save the actual position
        filename = self.dataset_name + "/" + self.dataset_name + "_obj_pos.csv"
        np.savetxt(filename, pos, delimiter=",")

        return pos

    def _set_obj_pos(self, obj_index, t):
        # set position
        self.model.body_pos[obj_index, 0] = self.obj_pos[t, 0]
        self.model.body_pos[obj_index, 1] = self.obj_pos[t, 1]

    def _randomise_light_pos(self):
        x = uniform(-5, 5)
        y = uniform(-5, 5)

        # set position
        # body_pos is hard coded for now
        self.model.light_pos[0, 0] = uniform(-10, 10)
        self.model.light_pos[0, 1] = uniform(-10, 5)

    def _save_pos_2_np(self, index, x, y):
        self.all_pos[index, 0] = x
        self.all_pos[index, 1] = y

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

    def _save_state(self, state):
        True


if __name__ == '__main__':
    sim = Simulator("xmls/box.xml", "testset_mj", offscreen=True, rand=False)
    cameras = [0]

    # preview model
    # sim.render()

    t0 = time.time()

    # create dataset
    sim.create_dataset(10, cameras)

    t1 = time.time()

    print(f"Time to complete: {t1-t0} seconds")
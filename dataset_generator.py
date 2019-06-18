# import mujoco_py as mj
# import os
#
# mj_path, _ = mj.utils.discover_mujoco()
# print(mj_path)
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mj.load_model_from_path(xml_path)
# sim = mj.MjSim(model)
#
# print(sim.data.qpos)
#
# sim.step()
# print(sim.data.qpos


import mujoco_py
from mujoco_py.modder import TextureModder
import math
import os
import scipy.misc


class Simulator():

    def __init__(self, model_path, dataset_name):
        self.dataset_name = dataset_name
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.offscreen = mujoco_py.MjRenderContextOffscreen(self.sim)
        self.modder = TextureModder(self.sim)

    def create_dataset(self, step, cameras):
        self._make_dir()
        t = 0

        while True:
            self.sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
            self.sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
            t += 1
            self.sim.step()
            self.offscreen.render(420, 380)

            for name in self.sim.model.geom_names:
                self.modder.rand_all(name)

            temp = self.sim.get_state()
            for cam in cameras:
                self.offscreen.render(1920, 1080, cam)
                rgb = self.offscreen.read_pixels(1920, 1080)[0][::-1, :, :]
                self._save_fig_to_dir(rgb, t, cam)

            if t == step or os.getenv('TESTING') is not None:
                break

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
    sim = Simulator("xmls/fetch/main.xml", "testset")
    cameras = [-1, 1]
    sim.create_dataset(10, cameras)

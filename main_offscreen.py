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

    def __init__(self, model_path):
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.offscreen = mujoco_py.MjRenderContextOffscreen(self.sim, 0)
        self.modder = TextureModder(self.sim)

    def create_dataset(self, step):
        t = 0

        while True:
            self.sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
            self.sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
            t += 1
            self.sim.step()
            self.offscreen.render(420, 380)

            for name in self.sim.model.geom_names:
                self.modder.rand_all(name)

            temp = sim.get_state()
            self.offscreen.render(1920, 1080, 0)
            rgb = self.offscreen.read_pixels(1920, 1080)[0]
            scipy.misc.imsave('array.png', rgb)

            if t == step or os.getenv('TESTING') is not None:
                break

    def _save_pic(self, path):



if __name__ == '__main__':
    sim = Simulator("xmls/fetch/main.xml")
    sim.create_dataset(10);

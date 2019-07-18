import mujoco_py
from mujoco_py.modder import TextureModder
import math
import random
import os
import scipy.misc
from matplotlib import pyplot
import numpy as np

model = mujoco_py.load_model_from_path("xmls/box.xml")

sim = mujoco_py.MjSim(model)
print("number of texture {}", sim.model.ntex)
viewer = mujoco_py.MjViewer(sim)
modder = TextureModder(sim)
modder.whiten_materials()
t = 0

# An array that is 0.1 apart
sample_values = np.arange(0, 1, 0.1)



while True:
    # Change actuator input
    # sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    # sim.data.ctrl[1] = math.sin(t / 10.) * 0.01

    # Change box position randomly
    # model.body_pos[1, 0] = random.uniform(-1, 1)
    # model.body_pos[1, 1] = random.uniform(-1, 1)
    # model.body_pos[1, 2] = random.uniform(-1, 1)

    # Change light box position randomly
    # model.light_pos[0, 0] = random.uniform(-10, 10)
    # model.light_pos[0, 1] = random.uniform(-10, 10)

    # Change texture
    # for name in sim.model.geom_names:
    #     modder.rand_all(name)

    # Change opengl parameters
    # sim.model.mat_emission[1] = sample_values[t % 10]
    # sim.model.mat_specular[1] = sample_values[t % 10]
    sim.model.mat_shininess[1] = sample_values[t % 10]
    # sim.model.mat_reflectance[1] = sample_values[t % 10]
    # print(sample_values[t % 10])


    t += 1
    sim.step()
    viewer.render()

    if t == 100 or t == 200:
        temp = sim.get_state()
        # rgb = viewer._read_pixels_as_in_window()
        # rgb = sim.render(width=1024, height=1024, camera_name="external_camera_0")
        # scipy.misc.imsave('array.png', rgb)
        # pyplot.imshow(rgb)
        # pyplot.show()

    if t > 100 and os.getenv('TESTING') is not None:
        break


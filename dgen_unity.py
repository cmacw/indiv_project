import os
from mjremote import mjremote
import time
from random import uniform
import numpy as np
from matplotlib import pyplot
import pickle
import dgen_mujoco
import scipy


class Simulator:
    def __init__(self, address, port, dataset_name):
        self.m_remote = mjremote()
        self.dataset_name = dataset_name
        # Connect to the unity executable

    def create_dataset(self):
        self._make_dir()

        # Connect to the unity executable
        print('Connect: ', self.m_remote.connect(address=address, port=port))

        # Buffer for the image array
        b = bytearray(3 * self.m_remote.width * self.m_remote.height)
        for i in range(0, 10):
            # Set random position for the joints
            # rand_pos = [uniform(-5, 5), uniform(-5, 5)]
            # m_remote.setqpos(np.array(rand_pos))

            # Set the camera
            self.m_remote.setcamera(1)

            # Save the screen shot to the buffer b
            self.m_remote.getimage(b)

            # Save the image
            buf = np.reshape(b, (self.m_remote.height, self.m_remote.width, 3))[::-1, :, :]
            scipy.misc.imsave(self.dataset_name + "/trial" + str(i) + ".png", buf)
            # pyplot.imshow(rgb)
            # pyplot.show()

        self.m_remote.close()

    def _make_dir(self):
        try:
            os.mkdir(self.dataset_name)
            print("Directory " + self.dataset_name + " created")
        except FileExistsError:
            print("Directory " + self.dataset_name + " already created")

        print("Using " + self.dataset_name + " to store the dataset")


if __name__ == '__main__':
    os.chdir("datasets")

    # address and port specify in the unity plugin
    address = "127.0.0.1"
    port = 1050

    # Begin timer
    t0 = time.time()

    # Create the dataset
    sim = Simulator(address, port, "random_ut")
    sim.create_dataset()

    # Stop timer
    t1 = time.time()

    print('Time taken: ', (t1 - t0))


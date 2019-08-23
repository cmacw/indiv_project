import os
from mjremote import mjremote
import time
from random import uniform
import numpy as np
from matplotlib import pyplot
import pickle
import dgen_mujoco
import scipy

from DataSetGenerator import DataSetGenerator


class UnDataSetGenerator(DataSetGenerator):
    def __init__(self, address, port, dataset_name, cam_pos_file):
        super().__init__(dataset_name, cam_pos_file=cam_pos_file)
        self.m_remote = mjremote()
        self.address = address
        self.port = port

    def create_data_set(self, ndata, radius_range, deg_range, cameras, start=0):
        self._make_dir()

        # initialise the camera position array
        self.cam_pos = self._get_cam_pos(radius_range, deg_range, 0.2, ndata)

        # Connect to the unity executable
        print('Connect: ', self.m_remote.connect(address=address, port=port))

        t = start

        # Buffer for the image array
        b = bytearray(3 * self.m_remote.width * self.m_remote.height)

        # Set the camera
        self.m_remote.setcamera(0)
        while True:
            self.m_remote.setcamposrot(self.cam_pos[t, :], b)

            # Save the screen shot to the buffer b
            self.m_remote.getimage(b)

            # Save the image
            rgb = np.reshape(b, (self.m_remote.height, self.m_remote.width, 3))[::-1, :, :]
            # TODO: find out cam id function in Unity
            cam_id = 0
            self._save_fig_to_dir(rgb, t, cam_id)
            # pyplot.imshow(rgb)
            # pyplot.show()

            # Time advance one
            t += 1

            # Print progress to terminal
            self.print_progress(ndata, t)

            if t == ndata or os.getenv('TESTING') is not None:
                print("Finish creating {} {} images".format(ndata, self.data_set_name))
                break

        self.m_remote.close()


if __name__ == '__main__':
    os.chdir("datasets/Set05")

    # address and port specify in the unity plugin
    address = "127.0.0.1"
    port = 1050

    # Begin timer
    t0 = time.time()

    # Create the dataset
    cameras = ["camera1"]
    sim = UnDataSetGenerator(address, port, "random_un_test", "cam_pos_test.csv")
    sim.create_data_set(5000, [0.25, 0.7], [-15, 15], cameras)

    # Stop timer
    t1 = time.time()

    print('Time taken: ', (t1 - t0))

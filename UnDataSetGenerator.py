import os
from mjremote import mjremote
import time
from random import uniform
import numpy as np
from matplotlib import pyplot
import pickle
import MjDataSetGenerator
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
        print('Connect: ', self.m_remote.connect(address=self.address, port=self.port))

        t = start

        # Buffer for the image array
        b = bytearray(3 * self.m_remote.width * self.m_remote.height)

        # Set the camera
        cam_id = 0
        self.m_remote.setcamera(cam_id)
        while True:
            # dummy_pos = np.array([0, -0.075, 0.0145, 1, 0, 0, 0, 0, -1, 0, 1,
            #                       0])  # 0, 0.680768502606714, -0.732498631984124, 0, 0.732498631984124,  0.680768502606714])
            # self.m_remote.setcamposrot(dummy_pos, b)

            self.m_remote.setcamposrot(self.cam_pos[t, :], b)

            # Save the screen shot to the buffer b
            self.m_remote.getimage(b)

            # Save the image
            rgb = np.reshape(b, (self.m_remote.height, self.m_remote.width, 3))[::-1, :, :]
            # TODO: find out cam id function in Unity
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
    os.chdir("datasets/Set06")

    # address and port specify in the unity plugin
    address = "127.0.0.1"
    port = 1050

    cameras = ["targetcam"]

    # Create the dataset
    # Begin timer
    t0 = time.time()
    sim = UnDataSetGenerator(address, port, "realistic_un_valid", "cam_pos_valid.csv")
    sim.create_data_set(1000, [0.07, 0.5], [-100, -80], cameras)
    # Stop timer
    t1 = time.time()
    print('Time taken: ', (t1 - t0))

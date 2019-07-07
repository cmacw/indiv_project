from mjremote import mjremote
import time
from random import uniform
import numpy as np
from matplotlib import pyplot
import pickle
import dataset_generator_mujoco
import scipy

m_remote = mjremote()

# address and port specify in the unity plugin
address = "127.0.0.1"
port = 1050

# Connect to the unity executable
print('Connect: ', m_remote.connect(address=address, port=port))

# Buffer for the image array
b = bytearray(3 * m_remote.width * m_remote.height)
# b = bytearray(3 * 1920 * 1080)  #attempt to increase resolution: UNSUCCESSFUL

# Time for calculating fps
t0 = time.time()

# Set the camera
m_remote.setcamera(1)

# Main loop
t0 = time.time()

for i in range(0, 10):
    # Set random position for the joints
    # rand_pos = [uniform(-5, 5), uniform(-5, 5)]
    # m_remote.setqpos(np.array(rand_pos))

    # Save the screen shot to the buffer b
    m_remote.getimage(b)

    # Save the image
    buf = np.reshape(b, (m_remote.height, m_remote.width, 3))[::-1, :, :]
    scipy.misc.imsave("testdata_unity/trial"+str(i)+".png", buf)
    # pyplot.imshow(rgb)
    # pyplot.show()

# Print the fps
t1 = time.time()
print('Time taken: ', (t1 - t0))
m_remote.close()

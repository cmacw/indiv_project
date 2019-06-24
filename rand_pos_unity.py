from mjremote import mjremote
import time
from random import uniform,randint
import numpy as np
from matplotlib import pyplot
import pickle
import dataset_generator
import scipy

m = mjremote()

# address and port specify in the unity plugin
address = "127.0.0.1"
port = 1050


print('Connect: ', m.connect(address = address, port = port))
b = bytearray(3 * m.width * m.height)
t0 = time.time()
for i in range(0, 100):
    rand_pos = [uniform(-1, 1), uniform(-1, 1)]
    m.setqpos(np.array(rand_pos))
    m.getimage(b)

    if i == 50:
        m.setcamera(1)
        rgb = np.reshape(b, (m.height, m.width, 3))[::-1, :, :]
        scipy.misc.imsave("trial1.png", rgb)
        # pyplot.imshow(rgb)
        # pyplot.show()
        m.setcamera(-1)
        m.getimage(b)
        rgb = np.reshape(b, (m.height, m.width, 3))[::-1, :, :]
        scipy.misc.imsave("trial2.png", rgb)



t1 = time.time()
print('FPS: ', 100 / (t1 - t0))
m.close()

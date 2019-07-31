import mujoco_py
from mujoco_py.modder import TextureModder, CameraModder
import os
import time
from random import uniform

from DataSetGenerator import DataSetGenerator


class MjDataSetGenerator(DataSetGenerator):
    IMG_SIZE = 512

    def __init__(self, model_path, dataset_name, rand=False, cam_pos_file=None, cam_norm_pos_file=None):
        super().__init__(dataset_name, cam_pos_file=cam_pos_file, cam_norm_pos_file=cam_norm_pos_file)
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None)
        self.tex_modder = TextureModder(self.sim) if rand else None
        self.cam_modder = CameraModder(self.sim) if rand else None

    def on_screen_render(self, cam_name):
        self.sim.reset()
        temp_viewer = mujoco_py.MjViewer(self.sim)
        t = 0

        while True:
            # Randomised material/texture if any is assigned to geom
            self._rand_mat()

            # Set camera position and orientation
            self._set_cam_pos(cam_name, t)
            self.sim.step()
            self._set_cam_orientation(cam_name, t)

            temp_viewer.render()
            t += 1
            if t > 100 and os.getenv('TESTING') is not None:
                break

    def create_data_set(self, ndata, radius_range, deg_range, quat, cameras, start=0):
        self.sim.reset()
        self._make_dir()

        t = start

        # initialise the camera position array
        self.cam_pos = self._get_cam_pos(radius_range, deg_range, quat, ndata)

        # generate dataset
        while True:

            # Randomised the position of the object
            # Set camera position
            self._set_cam_pos(cameras[0], t)

            # Randomised light source position
            self._randomise_light_pos()

            # Randomised material/texture if any is assigned to geom
            self._rand_mat()

            # Simulate and render in offscreen renderer
            self.sim.step()

            # Save images for all camera
            for cam in cameras:
                self._set_cam_orientation(cam, t)
                cam_id = self.cam_modder.get_camid(cam)
                self.viewer.render(self.IMG_SIZE, self.IMG_SIZE, cam_id)
                rgb = self.viewer.read_pixels(self.IMG_SIZE, self.IMG_SIZE)[0][::-1, :, :]
                self._save_fig_to_dir(rgb, t, cam_id)

            # Time advance one
            t += 1

            # Print progress to terminal
            self.print_progress(ndata, t)

            if t == ndata or os.getenv('TESTING') is not None:
                print("Finish creating {} {} images".format(ndata, self.data_set_name))
                break

        self._save_cam_pos(self.cam_pos)

    def _rand_mat(self):
        if self.tex_modder is not None:
            for name in self.sim.model.geom_names:
                geom_id = self.model.geom_name2id(name)
                mat_id = self.model.geom_matid[geom_id]
                if mat_id >= 0:
                    self.tex_modder.rand_all(name)

    def _set_cam_pos(self, cam_name, t, printPos=None):
        # If no
        if self.cam_pos is None:
            self.cam_pos = self._get_cam_pos([0.25, 0.7], [0, 80], 0.5)

        # set position of the reference camera
        cam_id = self.cam_modder.get_camid(cam_name)
        self.model.cam_pos[cam_id] = self.cam_pos[t, 0:3]

        if printPos:
            print("The cam pos is: ", self.cam_pos[t, :])

    # Call after sim.step if want to change the camera orientation while keep

    def _set_cam_orientation(self, cam_name, t, printPos=None):
        cam_id = self.cam_modder.get_camid(cam_name)
        if self.cam_pos_file is None:
            self.sim.data.cam_xmat[cam_id] = self.sim.data.cam_xmat[cam_id] + self.cam_pos[t, 3:]
            self.cam_pos[t, 3:] = self.sim.data.cam_xmat[cam_id]
        else:
            self.sim.data.cam_xmat[cam_id] = self.cam_pos[t, 3:]

        if printPos:
            print("The cam orientation is: ", self.cam_pos[t, :])

    def _randomise_light_pos(self):
        x = uniform(-5, 5)
        y = uniform(-5, 5)

        # set position
        # body_pos is hard coded for now
        self.model.light_pos[0, 0] = uniform(-10, 10)
        self.model.light_pos[0, 1] = uniform(-10, 5)


if __name__ == '__main__':
    os.chdir("datasets")
    sim = MjDataSetGenerator("../xmls/box.xml", "random_mj", cam_pos_file="cam_pos.csv", rand=True)

    # preview model
    # sim.on_screen_render("targetcam")

    t0 = time.time()

    # create dataset
    cameras = ["targetcam"]
    # TODO: change the argument so if cam_pos_file is present, no other arguments are needed
    sim.create_data_set(100, [0.25, 0.7], [0, 80], 0.5, cameras)

    t1 = time.time()

    print(f"Time to complete: {t1 - t0} seconds")

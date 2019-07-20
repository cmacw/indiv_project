import mujoco_py
from mujoco_py.modder import TextureModder, CameraModder
import os
import time

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
            # Randomised material/texture if required
            if self.tex_modder is not None:
                for name in self.sim.model.geom_names:
                    self.tex_modder.rand_all(name)

            # Set camera position and orientation
            self._set_cam_pos(cam_name, t)
            self.sim.step()
            self._set_cam_orientation(cam_name, t)

            temp_viewer.render()
            t += 1
            if t > 100 and os.getenv('TESTING') is not None:
                break

    def create_dataset(self, ndata, radius_range, deg_range, quat, cameras, start=0):
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

            # Randomised material/texture if required
            if self.tex_modder is not None:
                for name in self.sim.model.geom_names:
                    self.tex_modder.rand_all(name)

            # Simulate and render in offscreen renderer
            self.sim.step()

            # Save images for all camera
            for cam in cameras:
                self._set_cam_orientation(cam, t)
                cam_id = self.cam_modder.get_camid(cam)
                self.viewer.render(self.IMG_SIZE, self.IMG_SIZE, cam_id)
                rgb = self.viewer.read_pixels(self.IMG_SIZE, self.IMG_SIZE)[0][::-1, :, :]
                self._save_fig_to_dir(rgb, t, cam_id)

            t += 1
            # Print progress
            if t % 100 == 0:
                print("Progress: {} / {}".format(t, ndata))

            if t == ndata or os.getenv('TESTING') is not None:
                print("Finish creating {} {} images".format(ndata, self.data_set_name))
                break


if __name__ == '__main__':
    os.chdir("datasets")
    sim = MjDataSetGenerator("../xmls/box.xml", "trial", cam_pos_file="cam_pos.csv", rand=True)

    # preview model
    # sim.on_screen_render("targetcam")

    t0 = time.time()

    # create dataset
    cameras = ["targetcam"]
    sim.create_dataset(50, [0.25, 0.7], [0, 80], 0.5, cameras)

    t1 = time.time()

    print(f"Time to complete: {t1 - t0} seconds")

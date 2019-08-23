import os
import time
from random import uniform

import mujoco_py
from mujoco_py.modder import TextureModder, CameraModder

from DataSetGenerator import DataSetGenerator


class MjDataSetGenerator(DataSetGenerator):
    IMG_SIZE = 128

    def __init__(self, model_path, dataset_name, use_procedural=False, cam_pos_file=None, cam_norm_pos_file=None):
        super().__init__(dataset_name, cam_pos_file=cam_pos_file, cam_norm_pos_file=cam_norm_pos_file)
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None)
        self.cam_modder = CameraModder(self.sim)
        self.tex_modder = TextureModder(self.sim)
        self.use_procedural = use_procedural

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

    def create_data_set(self, ndata, radius_range, deg_range, quat,
                        cameras, target_geom, png_tex_ids=None, start=0):
        self.sim.reset()
        self._make_dir()

        t = start

        # initialise the camera position array
        self.cam_pos = self._get_cam_pos(radius_range, deg_range, quat, ndata, start)
        # self.cam_pos = self._get_debug_cam_pos(radius_range, deg_range, quat, ndata, 4)

        # generate dataset
        while True:

            # Randomised the position of the object
            # Set camera position
            self._set_cam_pos(cameras[0], t)

            # Randomised light source position
            self._randomise_light_pos()

            # Randomised material/texture if any is assigned to geom
            self._rand_mat(target_geom, png_tex_ids)

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

    def _rand_mat(self, target_geom, png_tex):
        for name in self.sim.model.geom_names:
                # Random mat using mujoco py modder or png in xml
                if self.use_procedural:
                    self.tex_modder.rand_all(name)
                else:
                    # change redness
                    if name == target_geom["cube"]:
                        self._change_redness(name)

                    # change wood texture
                    if name == target_geom["ground"]:
                        self._change_tex_png(name, png_tex)

    def _get_tex_id(self, mat_id):
        tex_id = self.model.mat_texid[mat_id]
        assert tex_id >= 0, "Material has no assigned texture"
        return tex_id

    def _change_redness(self, name):
        r = self.tex_modder.random_state.randint(120, 256)
        g = self.tex_modder.random_state.randint(0, 120)
        b = self.tex_modder.random_state.randint(0, 120)
        self.tex_modder.set_rgb(name, [r, g, b])

    def _change_tex_png(self, name, png_tex):
        geom_id = self.model.geom_name2id(name)
        mat_id = self.model.geom_matid[geom_id]
        self.model.mat_texid[mat_id] = \
            self.tex_modder.random_state.randint(png_tex[0], png_tex[1] + 1)
        self.tex_modder.upload_texture(name)

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
            # Add the offset to the rotational matrix
            self.sim.data.cam_xmat[cam_id] = self.sim.data.cam_xmat[cam_id] + self.cam_pos[t, 3:]
            # Save the actual rotational matrix
            self.cam_pos[t, 3:] = self.sim.data.cam_xmat[cam_id]
        else:
            self.sim.data.cam_xmat[cam_id] = self.cam_pos[t, 3:]

        if printPos:
            print("The cam orientation is: ", self.cam_pos[t, :])

    def _randomise_light_pos(self):
        # set position
        # index of light is hard coded for now
        # TODO: get light index by name
        self.model.light_pos[0, 0] = uniform(-10, 10)
        self.model.light_pos[0, 1] = uniform(-10, 5)


if __name__ == '__main__':
    os.chdir("datasets/Set05")
    sim = MjDataSetGenerator("../../xmls/box.xml", "random_mj",
                             use_procedural=True, cam_pos_file="cam_pos.csv")

    # preview model
    # sim.on_screen_render("targetcam")

    t0 = time.time()

    # create dataset
    cameras = ["targetcam"]
    target_geom = {"cube": "boxgeom", "ground": "ground"}
    png_tex_ids = (5, 15)

    # TODO: change the argument so if cam_pos_file is present, no other arguments are needed
    sim.create_data_set(50000, [0.25, 0.7], [-10, 10], 0.1, cameras, target_geom, png_tex_ids)

    t1 = time.time()

    print(f"Time to complete: {t1 - t0} seconds")

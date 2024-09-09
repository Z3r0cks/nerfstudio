# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This file contains the render state machine, which is responsible for deciding when to render the image """
from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, get_args

import numpy as np
import torch
from viser import ClientHandle
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils import colormaps, writer
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.utils import CameraState, get_camera
from nerfstudio.viewer_legacy.server import viewer_utils

#-------------------------------------------------------------
from nerfstudio.utils.debugging import Debugging
from scipy.spatial.transform import Rotation as R
import viser.transforms as vtf
import math
import csv
import json
#-------------------------------------------------------------
VISER_NERFSTUDIO_SCALE_RATIO: float = 1.0

if TYPE_CHECKING:
    from nerfstudio.viewer.viewer import Viewer
    from nerfstudio.viewer.viewer_density import ViewerDensity

RenderStates = Literal["low_move", "low_static", "high"]
RenderActions = Literal["rerender", "move", "static", "step"]

@dataclass
class RenderAction:
    """Message to the render state machine"""

    action: RenderActions
    """The action to take """
    camera_state: CameraState
    """The current camera state """


class RenderStateMachine(threading.Thread):
    """The render state machine is responsible for deciding how to render the image.
    It decides the resolution and whether to interrupt the current render.

    Args:
        viewer: the viewer state
    """

    def __init__(self, viewer: ViewerDensity, viser_scale_ratio: float, client: ClientHandle):
        threading.Thread.__init__(self)
        self.transitions: Dict[RenderStates, Dict[RenderActions, RenderStates]] = {
            s: {} for s in get_args(RenderStates)
        }
        # by default, everything is a self-transition
        for a in get_args(RenderActions):
            for s in get_args(RenderStates):
                self.transitions[s][a] = s
        # then define the actions between states
        self.transitions["low_move"]["static"] = "low_static"
        self.transitions["low_static"]["static"] = "high"
        self.transitions["low_static"]["step"] = "high"
        self.transitions["low_static"]["move"] = "low_move"
        self.transitions["high"]["move"] = "low_move"
        self.transitions["high"]["rerender"] = "low_static"
        self.next_action: Optional[RenderAction] = None
        self.state: RenderStates = "low_static"
        self.render_trigger = threading.Event()
        self.target_fps = 30
        self.viewer = viewer
        self.interrupt_render_flag = False
        self.daemon = True
        self.output_keys = {}
        self.viser_scale_ratio = viser_scale_ratio
        self.client = client
        self.running = True
        
        #-------------------------------------------------------------
        self.mesaurement_point_coordinates = []
        self.ray_id = 0
        self.side_id = 0
        self.density_threshold = 0
        self.fov_x = 15
        self.fov_y = 15
        self.width = 10
        self.height = 10
        self.mesh_objs = []
        self.mesaurement_conversion = None
        self.gui_button = self.viewer.viser_server.add_gui_button("LiDAR GUI", color="blue").on_click(lambda _: self.generate_lidar_gui())
        self.dataparser_transforms = {'transform': [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]}
        
        try:
            with open(self.viewer.dataparser_transforms_path) as f:
                dataparser_transforms = json.load(f)
        except:
            print("No dataparser_transforms.json found")
        
        self.translate_pos_from_omnivers = (dataparser_transforms["transform"][0][3], dataparser_transforms["transform"][1][3], dataparser_transforms["transform"][2][3])
        self.x_omni, self.y_omni, self.z_omni = self.translate_pos_from_omnivers

    def action(self, action: RenderAction):
        """Takes an action and updates the state machine

        Args:
            action: the action to take
        """
        if self.next_action is None:
            self.next_action = action
        elif action.action == "step" and (self.state == "low_move" or self.next_action.action in ("move", "rerender")):
            # ignore steps if:
            #  1. we are in low_moving state
            #  2. the current next_action is move, static, or rerender
            return
        elif self.next_action.action == "rerender":
            # never overwrite rerenders
            pass
        elif action.action == "static" and self.next_action.action == "move":
            # don't overwrite a move action with a static: static is always self-fired
            return
        else:
            #  monimal use case, just set the next action
            self.next_action = action

        # handle interrupt logic
        if self.state == "high" and self.next_action.action in ("move", "rerender"):
            self.interrupt_render_flag = True
        self.render_trigger.set()

    def _render_img(self, camera_state: CameraState):
        """Takes the current camera, generates rays, and renders the image

        Args:
            camera_state: the current camera state
        """
        # initialize the camera ray bundle
        if self.viewer.control_panel.crop_viewport:
            obb = self.viewer.control_panel.crop_obb # oriented bounding box
        else:
            obb = None

        image_height, image_width = self._calculate_image_res(camera_state.aspect)
        # These 2 lines make the control panel's time option independent from the render panel's.
        # When outside of render preview, it will use the control panel's time.
        if not self.viewer.render_tab_state.preview_render and self.viewer.include_time:
            camera_state.time = self.viewer.control_panel.time
        camera = get_camera(camera_state, image_height, image_width)        
        camera = camera.to(self.viewer.get_model().device) # cuda:0
        assert isinstance(camera, Cameras)
        assert camera is not None, "render called before viewer connected"

        with TimeWriter(None, None, write=False) as vis_t:
            with self.viewer.train_lock if self.viewer.train_lock is not None else contextlib.nullcontext():

                self.viewer.get_model().eval()
                step = self.viewer.step
                try:
                    with torch.no_grad(), viewer_utils.SetTrace(self.check_interrupt):
                        outputs = self.viewer.get_model().get_outputs_for_camera(camera, obb_box=obb)
                except viewer_utils.IOChangeException:
                    self.viewer.get_model().train()
                    raise
                self.viewer.get_model().train()
            num_rays = (camera.height * camera.width).item()
            
            if self.viewer.control_panel.layer_depth:
                R = camera.camera_to_worlds[0, 0:3, 0:3].T
                camera_ray_bundle = camera.generate_rays(camera_indices=0, obb_box=obb)
                pts = camera_ray_bundle.directions * outputs["depth"]
                pts = (R @ (pts.view(-1, 3).T)).T.view(*camera_ray_bundle.directions.shape)
                outputs["gl_z_buf_depth"] = -pts[..., 2:3]  # negative z axis is the coordinate convention
        render_time = vis_t.duration
        if writer.is_initialized() and render_time != 0:
            writer.put_time(
                name=EventName.VIS_RAYS_PER_SEC, duration=num_rays / render_time, step=step, avg_over_steps=True
            )
        return outputs

    def run(self):
        """Main loop for the render thread"""
        
        while self.running:
            if not self.viewer.ready:
                time.sleep(0.1)
                continue
            if not self.render_trigger.wait(0.2):
                # if we haven't received a trigger in a while, send a static action
                self.action(RenderAction(action="static", camera_state=self.viewer.get_camera_state(self.client)))
            action = self.next_action
            self.render_trigger.clear()
            if action is None:
                continue
            self.next_action = None
            if self.state == "high" and action.action == "static":
                # if we are in high res and we get a static action, we don't need to do anything
                continue
            self.state = self.transitions[self.state][action.action]
            try:
                outputs = self._render_img(action.camera_state)

            except viewer_utils.IOChangeException:
                # if we got interrupted, don't send the output to the viewer
                continue
            

            self._send_output_to_viewer(outputs, static_render=(action.action in ["static", "step"]))

    def mark_nearby(self, tree, point, distance_threshold):
        return tree.query_ball_point(point, r=distance_threshold, p=2)

    def filter_nearby_indices(self, points, distance_threshold=0.003, n_jobs=4):
        """Returns indices of points that are not removed due to proximity"""
        from scipy.spatial import cKDTree
        from concurrent.futures import ThreadPoolExecutor

        # Aufbau eines KD-Baums für schnelle Nachbarschaftssuche
        tree = cKDTree(points)
        kept_indices = np.ones(len(points), dtype=bool)

        def process_point(i):
            if kept_indices[i]:
                # Suche alle nahen Punkte, die innerhalb des Schwellenwerts liegen
                nearby_indices = self.mark_nearby(tree, points[i], distance_threshold)
                kept_indices[nearby_indices] = False
                kept_indices[i] = True

        # Parallelverarbeitung der Punkte mit ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            list(executor.map(process_point, range(len(points))))

        # Rückgabe der Indizes der verbleibenden Punkte
        return np.where(kept_indices)[0]
    
    def delete_measurement_point(self, id):
        print(self.mesaurement_point_coordinates)
        try:
            if id == 1:
                    self.mesaurement_point_coordinates[0] = None
                    print("No point 1 set")
            elif id == 2:
                    self.mesaurement_point_coordinates[1] = None
                    print("No point 2 set") 
            elif id == 3:
                    self.mesaurement_conversion = None
                    print("No Converstion set") 
        except:
            print("Cannot remove")
    
    def generate_lidar_gui(self) -> None:
        '''Add GUI elements for LiDAR'''
        
        viser = self.viewer.viser_server        
        #TODO: explain path in documentation
        self.perspectiv = viser.add_camera_frustum(name="perspectiv", fov=5.0, aspect=1, scale=0.1, color=(235, 52, 79), wxyz=(1, 0, 0, 0), position=(self.x_omni, self.y_omni, self.z_omni))
        
        with viser.add_gui_folder("Calibration"):
            viser.add_gui_button("Measure Point 1", color="blue").on_click(lambda _: self._show_density(measure=[True, 0]))
            viser.add_gui_button("Measure Point 2", color="blue").on_click(lambda _: self._show_density(measure=[True, 1]))
            self.m_point_1 = viser.add_gui_text("Coord Measurement Point 1", "Not Set")
            self.m_point_2 = viser.add_gui_text("Coord Measurement Point 2", "Not Set")
            viser.add_gui_text("Conversion", "Not Set")
            
            with viser.add_gui_folder("Correction", expand_by_default=False):
                viser.add_gui_button("Delete Point 1", color="red").on_click(lambda _: self.delete_measurement_point(1))
                viser.add_gui_button("Delete Point 2", color="red").on_click(lambda _: self.delete_measurement_point(2))
                viser.add_gui_button("Delete Conversion", color="red").on_click(lambda _: self.delete_measurement_point(3))
                
        with viser.add_gui_folder("Perspectiv Position"):
            self.perspectiv_pos_x = viser.add_gui_slider("Pos X", -20, 20, 0.01, 0)
            self.perspectiv_pos_y = viser.add_gui_slider("Pos Y", -20, 20, 0.01, 0)
            self.perspectiv_pos_z = viser.add_gui_slider("Pos Z (Height)", -20, 20, 0.01, 0)
        
        with viser.add_gui_folder("Perspectiv Orientation"):
            self.perspectiv_wxyz_x = viser.add_gui_slider("Rot X", -180, 180, 0.1, 0)
            self.perspectiv_wxyz_y = viser.add_gui_slider("Rot Y", -180, 180, 0.1, 0)
            self.perspectiv_wxyz_z = viser.add_gui_slider("Rot Z", -180, 180, 0.1, 0)
        
        with open('../nerfstudio/lidar_settings.json') as f:
            lidar_data = json.load(f)
            
        self.h_angle_resolution_dropdown = viser.add_gui_dropdown("Horizontal Resolution", ["0.125", "0.25", "0.5", "1", "2", "3", "4", "5"], "1")
        self.v_angle_resolution_dropdown = viser.add_gui_dropdown("Vertical Angle Resolution", ["0.125", "0.25", "0.5", "1", "2", "3", "4", "5"], "1")
        
        for lidar in lidar_data:
            scanner_settings = lidar_data[lidar]
            with viser.add_gui_folder(lidar_data[lidar]["name"], expand_by_default=False):
                viser.add_gui_button("Generate Point Cloud", color="blue").on_click(lambda _, scanner_settings=scanner_settings: self._show_density(scanner_settings=scanner_settings))
                viser.add_gui_button("Generate Plot", color="green").on_click(lambda _, scanner_settings=scanner_settings: self._show_density(plot_density=True, scanner_settings=scanner_settings))
                viser.add_gui_button("Show Rays", color="pink").on_click(lambda _, scanner_settings=scanner_settings: self._show_density(debugging=True, scanner_settings=scanner_settings))
                viser.add_gui_button("Clear Point Cloud", color="red").on_click(lambda _: self.delete_point_cloud())

        with viser.add_gui_folder("Dev Options", expand_by_default=False):
            with viser.add_gui_folder("Camera Options", expand_by_default=False):
                viser.add_gui_button("Viser Camera To Perspectiv", color="violet").on_click(lambda _: self.set_perspectiv_camera("viser_perspectiv"))
                viser.add_gui_button("Perspectiv To Viser Camera", color="violet").on_click(lambda _: self.set_perspectiv_camera(""))
                
            with viser.add_gui_folder("Debugging", expand_by_default=False):
                viser.add_gui_button("Show all samples per ray", color="blue").on_click(lambda _: self._show_density(debugging=True))
                viser.add_gui_button("Print neares density", color="cyan").on_click(lambda _: self.get_ray_infos())
                viser.add_gui_button("Print single ray inf.", color="red").on_click(lambda _: self._scan_density())
                viser.add_gui_button("Take screenshot", color="green").on_click(lambda _: self.take_screenshot())
                
            with viser.add_gui_folder("Density Options"):
                viser.add_gui_button("Pointcloud", color="green").on_click(lambda _: self._show_density())
                viser.add_gui_button("Pointcloud Clickable (slow)", color="pink").on_click(lambda _: self._show_density(clickable=True))
                viser.add_gui_button("Plot Densites", color="indigo").on_click(lambda _: self._show_density(plot_density=True))
                viser.add_gui_button("Toggle Labels", color="cyan").on_click(lambda _: self.toggle_labels())
                viser.add_gui_button("Clear Point Cloud", color="red").on_click(lambda _: self.delete_point_cloud())
                
            with viser.add_gui_folder("Density Settings"):
                with viser.add_gui_folder("Width (X)"):
                    self.perspectiv_fov_x = viser.add_gui_slider("FOV Horizontal", 0, 360, 1, 30)
                    self.perspectiv_width = viser.add_gui_slider("Angular Resolution X", 1, 5000, 1, 2)
                with viser.add_gui_folder("Height (Y)"):
                    self.perspectiv_fov_y = viser.add_gui_slider("FOV Vertical", 0, 360, 1, 30)
                    self.perspectiv_heigth = viser.add_gui_slider("Angular Resolution Y", 1, 5000, 1, 1)
                
            with viser.add_gui_folder("ID Settings", expand_by_default=False):
                self.ray_id_slider = viser.add_gui_slider("Ray ID", 0, 10000, 1, 0)
                self.side_id_slider = viser.add_gui_slider("Side ID", 0, 10000, 1, 0)
                
            self.perspectiv_pos_x.on_update(lambda _: self.update_cube())
            self.perspectiv_pos_y.on_update(lambda _: self.update_cube())
            self.perspectiv_pos_z.on_update(lambda _: self.update_cube())
            self.perspectiv_wxyz_x.on_update(lambda _: self.update_cube())
            self.perspectiv_wxyz_y.on_update(lambda _: self.update_cube())
            self.perspectiv_wxyz_z.on_update(lambda _: self.update_cube())
            
            self.ray_id_slider.on_update(lambda _: setattr(self, "ray_id", self.ray_id_slider.value))
            self.side_id_slider.on_update(lambda _: setattr(self, "side_id", self.side_id_slider.value))
            
            self.perspectiv_fov_x.on_update(lambda _: setattr(self, "fov_x", self.perspectiv_fov_x.value))
            self.perspectiv_fov_y.on_update(lambda _: setattr(self, "fov_y", self.perspectiv_fov_y.value))
            self.perspectiv_heigth.on_update(lambda _: setattr(self, "height", self.perspectiv_heigth.value))
            self.perspectiv_width.on_update(lambda _: setattr(self, "width", self.perspectiv_width.value))

    def take_screenshot(self):
        import pyautogui
        # from PIL import Image
        
        file_name = "id_" + str(self.ray_id) + ".png"
        file_dir = "C:/Users/free3D/Desktop/Patrick_Kaserer/screenshots/single_ray_density/"
        screenshot = pyautogui.screenshot()
        
        crop_size = 300  # 50 pixels in all directions means a total of 100x100 pixels

        # Get the dimensions of the screenshot
        width, height = screenshot.size

        # Calculate the cropping box centered in the screenshot
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        cropped_image = screenshot.crop((left, top, right, bottom))
        cropped_image.save(file_dir + file_name)

    def get_ray_infos(self) -> None:
        for i in range(50):
            self._show_density(showNearesDensity=True)
        self.increment_ray_id()
    
    def increment_ray_id(self) -> None:
        self.ray_id += 1
        Debugging.log("ray_id: ", self.ray_id)
    
    def update_cube(self):
        self.perspectiv.wxyz = R.from_euler('xyz', [self.perspectiv_wxyz_x.value, self.perspectiv_wxyz_y.value, self.perspectiv_wxyz_z.value], degrees=True).as_quat()
        self.perspectiv.position = (self.perspectiv_pos_x.value + self.x_omni, self.perspectiv_pos_y.value + self.y_omni, self.perspectiv_pos_z.value + self.z_omni)

    def _scan_density(self) -> None:
            print_list = []
            for side in range(4):
                if side == 0:
                    self.perspectiv.position = 1 + self.x_omni, 1.1 + self.y_omni, 1.9 + self.z_omni
                elif side == 1:
                    self.perspectiv.position = 1 + self.x_omni, 1.1 + self.y_omni, 0.9 + self.z_omni
                elif side == 2:
                    self.perspectiv.position = 4 + self.x_omni, + 5.9 + self.y_omni, 1.9 + self.z_omni
                else:
                    self.perspectiv.position = 4 + self.x_omni, + 5.9 + self.y_omni, 0.9 + self.z_omni
                for horizont in range(8):
                    for vertical in range(8):
                        Rv = vtf.SO3(wxyz=self.perspectiv.wxyz)
                        Rv = Rv @ vtf.SO3.from_x_radians(np.pi)
                        Rv = torch.tensor(Rv.as_matrix())
                        origin = torch.tensor(self.perspectiv.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
                        c2w = torch.concatenate([Rv, origin[:, None]], dim=1)
                        
                        fx_value = self.width / (2 * math.tan(math.radians(self.fov_x / 2)))
                        fy_value = self.height / (2 * math.tan(math.radians(self.fov_x / 2)))

                        fx = torch.tensor([[fx_value]], device='cuda:0')
                        fy = torch.tensor([[fy_value]], device='cuda:0')
                        cx = torch.tensor([[self.width/2]], device='cuda:0')
                        cy = torch.tensor([[self.height/2]], device='cuda:0')

                        camera = Cameras(
                            camera_to_worlds=c2w,
                            fx=fx,
                            fy=fy,
                            cx=cx,
                            cy=cy,
                            width=torch.tensor([[self.width]]),
                            height=torch.tensor([[self.height]]),
                            distortion_params=None,
                            camera_type=torch.tensor([[1]], device='cuda:0'),
                            times=torch.tensor([[0.]], device='cuda:0')
                        )
                        assert isinstance(camera, Cameras)
                        outputs = self.viewer.get_model().get_outputs_for_camera(camera, width=self.width, height=self.height)

                        all_densities = []
                        all_density_locations = []
                        all_rgbs = []
                            
                        for densities, locations, rgb in zip(outputs["densities"], outputs["densities_locations"], outputs["rgb"]):
                            all_densities.append(densities)
                            all_density_locations.append(locations)
                            all_rgbs.append(rgb)

                        all_densities = torch.cat([ray_densities for ray_densities in all_densities if ray_densities.numel() > 0])
                        all_density_locations = torch.cat([ray_locations for ray_locations in all_density_locations if ray_locations.numel() > 0])
                        all_rgbs = torch.cat([ray_rgbs for ray_rgbs in all_rgbs if ray_rgbs.numel() > 0])
                        
                        for ray_locations, ray_densities, rgb in zip(all_density_locations, all_densities, all_rgbs):
                            for location, density in zip(ray_locations, ray_densities):
                                rgblist_normalized = rgb.tolist()
                                rgblist = [int(val * 255) for val in rgblist_normalized]
                                print_list.append([self.compute_distance(origin, location), density.item(), rgblist])
                            self.print_single_ray_informations(print_list)
                            print_list.clear()
                            self.ray_id += 1
                            x, y, z = self.perspectiv.position #type: ignore
                        self.perspectiv.position = x, y + 0.1, z #type: ignore
                    x, y, z = self.perspectiv.position #type: ignore
                    self.perspectiv.position = x, y - 0.8, z - 0.1
                self.side_id += 1
                
    def print_single_ray_informations(self, print_list):
        self.csv_filename = 'single_ray_informations.csv'
            # "side id" "id", "location", "distance", "density", rgb
        try:
            with open(self.csv_filename, 'x', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                headers = ["side_id", "ray_id", "distance", "density", "rgb"]
                csvwriter.writerow(headers)
        except FileExistsError:
            pass
        
        for i, info in enumerate(print_list):
            with open(self.csv_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                row = [self.side_id] + [self.ray_id] + info
                csvwriter.writerow(row)
        
    def _show_density(self, **kwargs) -> None:
        """Show the density in the viewer

        kwargs:
            plot_density: bool = False -> if the density should be plotted
            clickable: bool = False -> if every point should be clickable (very slow)
            showNearesDensity: bool = False -> show the nearesDenstiy to 1
            showSingelRayInf: bool = False -> Save the information of a single ray
            debugging: bool = False -> Show all points of the density
            scanner_settings: None | str = None -> the settings of the lidar scanner
        """
        
        print("Showing density")
        plot_density = None
        clickable = None
        showNearesDensity = None
        showSingelRayInf = None
        debugging = None
        scanner_settings = None
        measure = [False, 0]
        int_scanner_settings = {}
        
        for key, value in kwargs.items():
            if key == "plot_density":
                plot_density = value
            elif key == "clickable":
                clickable = value
            elif key == "showNearesDensity":
                showNearesDensity = value
            elif key == "measure":
                measure = value
            elif key == "showSingelRayInf":
                showSingelRayInf = value
            elif key == "debugging":
                debugging = value
            elif key == "scanner_settings":
                scanner_settings = value
                for key in scanner_settings:
                    if key == "name" or key == "description" or key == "angle_resolution":
                        continue
                    int_scanner_settings[key] = float(scanner_settings[key])
        
        Rv = vtf.SO3(wxyz=self.perspectiv.wxyz)
        Rv = Rv @ vtf.SO3.from_x_radians(np.pi)
        Rv = torch.tensor(Rv.as_matrix())
        origin = torch.tensor(self.perspectiv.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
        c2w = torch.concatenate([Rv, origin[:, None]], dim=1)
        
        if int_scanner_settings:
            if int_scanner_settings["vertical_opening_angel"] != 1:
                height = int(int_scanner_settings["vertical_opening_angel"] / float(self.v_angle_resolution_dropdown.value))
            else:
                height = 1
            if int_scanner_settings["horizontal_opening_angel"] != 1:
                width = int(int_scanner_settings["horizontal_opening_angel"] / float(self.h_angle_resolution_dropdown.value))
            else:
                width = 1
            fov_x = int_scanner_settings["horizontal_opening_angel"]
            fov_y = int_scanner_settings["vertical_opening_angel"]
        else:
            height = self.height
            width = self.width
            fov_x = self.fov_x
            fov_y = self.fov_y
        
        fx_value = width / (2 * math.tan(math.radians(fov_x / 2)))
        fy_value = height / (2 * math.tan(math.radians(fov_y / 2)))
        
        fx = torch.tensor([[fx_value]], device='cuda:0')
        fy = torch.tensor([[fy_value]], device='cuda:0')
        cx = torch.tensor([[width / 2]], device='cuda:0')
        cy = torch.tensor([[height / 2]], device='cuda:0')
        camera = Cameras(
            camera_to_worlds=c2w,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=torch.tensor([[width]]),
            height=torch.tensor([[height]]),
            distortion_params=None,
            camera_type=10,
            times=torch.tensor([[0.]], device='cuda:0')
        )
        
        assert isinstance(camera, Cameras)
        outputs = self.viewer.get_model().get_outputs_for_camera(camera, width=self.width, height=self.height)
        # Extrahiere die Dichtewerte und Dichtepositionen
        all_densities = []
        all_density_locations = []
            
        for densities, locations in zip(outputs["densities"], outputs["densities_locations"]):
            if densities.numel() > 0:
                all_densities.append(densities)
            if locations.numel() > 0:
                all_density_locations.append(locations)
                
        all_densities = torch.cat(all_densities)
        all_density_locations = torch.cat(all_density_locations)
                
        filtered_locations = []
        filtered_densities = []

        for ray_locations, ray_densities in zip(all_density_locations, all_densities):
            if ray_densities.numel() == 0:
                continue
            
            # takte the point which is nearest to nearestDistanceToCamera
            if showNearesDensity:
                distance_search = 1
                min_difference = float('inf')  # Initialisiere mit einem sehr großen Wert
                density_current = None
                location_current = None

                for location, density in zip(ray_locations, ray_densities):
                    distance = self.compute_distance(origin, location)
                    difference = abs(distance - distance_search)

                    if difference < min_difference:
                        min_difference = difference
                        density_current = density
                        location_current = location.tolist()
                        
                if density_current is not None and location_current is not None:
                    filtered_densities.append(density_current)
                    filtered_locations.append(location_current)
                    self.print_nearest_density(self.compute_distance(origin, location_current), density_current)
                    
            elif debugging:
                    for location, density in zip(ray_locations, ray_densities):
                        filtered_locations.append(location.tolist())   
                        filtered_densities.append(density) 
            else:
                # transmittance methode: -------------------------------------------------------------------------------------------
                distance, location, density = self.find_collision_with_transmittance(ray_locations, ray_densities)
                if distance is not None:
                    filtered_locations.append(location.tolist()) #type: ignore
                    filtered_densities.append(density.item()) #type: ignore
                       
        filtered_locations = torch.tensor(filtered_locations)
        filtered_densities = torch.tensor(filtered_densities)

        filtered_locations = filtered_locations.numpy()
        filtered_densities = filtered_densities.numpy()
        
        if plot_density:
            print("Plotting")
            import open3d as o3d
            # 3D point cloud visualisieren
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(filtered_locations)
            colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.ViewControl() # type: ignore
            o3d.visualization.draw_geometries([point_cloud]) # type: ignore
        
        print("clickable != None", clickable != None)
        
        if not plot_density and clickable == None and not showNearesDensity and not showSingelRayInf:
            print("Show Density")
            if len(self.mesh_objs) > 0:
                self.delete_point_cloud()
            
            Debugging.log("filtered_locations", filtered_locations.shape)
            # Debugging.log("filtered_locations", filtered_locations)
            obj = self.viewer.viser_server.add_point_cloud(name="density", points=filtered_locations*VISER_NERFSTUDIO_SCALE_RATIO, colors=(255, 0, 255), point_size=0.01, wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0), visible=True)
            self.mesh_objs.append(obj)
        
        if clickable != None or measure[0] != False:
            if len(self.mesh_objs) > 0:
                self.delete_point_cloud()
                
            for index, (location, density) in enumerate(zip(filtered_locations, filtered_densities)):
                self.add_point_as_mesh(location, index, density, measure=measure)
                
    # concept of transmittance T(t), describes the probability of a photon to pass through a medium without being absorbed
    # If the transmittance is low, the photon is more likely to be absorbed. Start with transmittance = 1 and multiply it with the transmittance of each point along the ray.
    
    def find_collision_with_transmittance(self, ray_locations, ray_densities, transmission_threshold=0.01):
        """
        Finds the collision point along a ray based on transmission values.
        
        ray_locations: Tensor of 3D coordinates representing points along the ray.
        ray_densities: Tensor of density values corresponding to each point.
        transmission_threshold: The threshold for the transmission probability to consider as a collision.
        
        returns: Distance from the origin to the collision point, the collision location and density.
        """
        total_distance = 0.0
        transmittance = 1.0  # Initial transmittance
        origin = ray_locations[0]

        for location, density in zip(ray_locations[1:], ray_densities[1:]):
            distance = self.compute_distance(origin, location)
            total_distance += distance
            delta_transmittance = torch.exp(-density * distance)
            transmittance *= delta_transmittance

            if transmittance < transmission_threshold:
                return total_distance, location, density

        return None, None, None  # No collision found
                
    def find_collision(self, ray_locations, ray_densities, threshold):
        """
        Finds the collision point along a ray based on density values.
        
        ray_locations: Tensor of 3D coordinates representing points along the ray.
        ray_densities: Tensor of density values corresponding to each point.
        threshold: Cumulative density value at which a collision is detected.
        
        returns: Distance from the origin to the collision point, the collision location and density.
        """
        cumulative_density = 0.0
        total_distance = 0.0
        origin = ray_locations[0]

        for location, density in zip(ray_locations[1:], ray_densities[1:]):
            distance = self.compute_distance(origin, location)
            total_distance += distance
            cumulative_density += density.item() * distance
            
            if cumulative_density >= threshold:
                return total_distance, location, density

        return None, None, None  # no collision found
                
    def add_point_as_mesh(self, location, index, density, scale_factor=1, base_size=0.01, color= (0, 0, 255), measure=[False, 0]):
        half_size = base_size / 2 * scale_factor 
        vertices = np.array([
            [location[0] * scale_factor - half_size, location[1] * scale_factor - half_size, location[2] * scale_factor],
            [location[0] * scale_factor + half_size, location[1] * scale_factor - half_size, location[2] * scale_factor],
            [location[0] * scale_factor + half_size, location[1] * scale_factor + half_size, location[2] * scale_factor],
            [location[0] * scale_factor - half_size, location[1] * scale_factor + half_size, location[2] * scale_factor]
        ])
        
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        
        mesh_name = f"location_{index}"
        obj = self.viewer.viser_server.add_mesh_simple(
            name=mesh_name,
            vertices=vertices,
            faces=faces,
            color=color,
            position=(0, 0, 0),  # position bereits in vertices definiert
            visible=True,
        )
        
        if measure[0]:
            obj.on_click(lambda _: self.add_distance_modal_between_points(location*VISER_NERFSTUDIO_SCALE_RATIO, obj, measure[1]))
        else:
            obj.on_click(lambda _: self.add_distance_modal(location*VISER_NERFSTUDIO_SCALE_RATIO, density))
        
        self.mesh_objs.append(obj)
        
    def print_nearest_density(self, distance, density):
        position = self.perspectiv.position.tolist() if isinstance(self.perspectiv.position, np.ndarray) else self.perspectiv.position
        rot = [self.perspectiv_pos_x.value, self.perspectiv_pos_y.value, self.perspectiv_pos_z.value]
        
        distance = float(distance) if isinstance(distance, float) else distance
        density = density.item() if isinstance(density, torch.Tensor) else density
        
        self.csv_filename = 'ray_information.csv'
        try:
            with open(self.csv_filename, 'x', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                headers = ["id", "position_x", "position_y", "position_z", "rotation_x", "rotation_y", "rotation_z", "distance", "density"]
                csvwriter.writerow(headers)
        except FileExistsError:
            pass
        
        print("id", type([self.ray_id]))
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            row = [self.ray_id] + position + rot + [distance, density]
            csvwriter.writerow(row)
            
    def add_distance_modal_between_points(self, coodinate, obj, point_id):
        point_button = self.m_point_2 if point_id == 1 else self.m_point_1
        if len(self.mesaurement_point_coordinates) == 1:
            (x1, y1, z1),(x2, y2, z2) = self.mesaurement_point_coordinates[0], self.mesaurement_point_coordinates[0]
            # x2, y2, z2 = self.mesaurement_point_coordinates[1]
            distance = self.compute_distance(self.mesaurement_point_coordinates[0], self.mesaurement_point_coordinates[1])
            distance_label = self.viewer.viser_server.add_label("distance_label", f"Distance: {distance:.4f}", (1, 0, 0, 0), ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2))
            
            vertices = np.array([self.mesaurement_point_coordinates[0], self.mesaurement_point_coordinates[1], self.mesaurement_point_coordinates[0]])
            faces = np.array([[0, 1, 2]])
            distance_ray = self.viewer.viser_server.add_mesh_simple(
                name="line_mesh",
                vertices=vertices,
                faces=faces,
                color=(155, 0, 0),
                wireframe=True, 
                opacity=None,
                material='standard',
                flat_shading=False,
                side='double',
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                visible=True
            )
            
            self.mesh_objs.append(distance_label)
            self.mesh_objs.append(distance_ray)
            distance_ray.on_click(lambda _: distance_ray.remove())
            self.set_measurement_point(point_button, point_id, coodinate, obj)
            # self.mesaurement_point_coordinates = []
        else:
            self.set_measurement_point(point_button, point_id, coodinate, obj)
            
    def set_measurement_point(self, point_button, point_id, coodinate, obj):
        self.mesaurement_point_coordinates.append(coodinate)
        Debugging.log("point_button._impl.value 1: ", point_button._impl.value)
        point_button._impl.__setattr__("value", f"{str(coodinate)}")
        obj.__setattr__("color", (255, 0, 0))

        # Debugging.log("api()", api)
        # Debugging.log("point_button._impl.value 2: ", point_button._impl.value)
        # Debugging.log("obj: ", obj)
        # Debugging.log("length of mesaurement_point_coordinates: ", len(self.mesaurement_point_coordinates))
        
    def add_distance_modal(self, point, density):
        """ 
        add a modal to show the distance of a point
        point: point
        """
        x, y, z = point
        global global_distance
        global global_density
        global_distance = self.compute_distance(self.perspectiv.position, point)
        global_density = density
        distance_label = self.viewer.viser_server.add_label("distance_label", f"Distance: {global_distance:.4f}", (1, 0, 0, 0), (x - 0.02, y, z + 0.04))
        # destity_label = self.viewer.viser_server.add_label("density_label", f"Density: {density:.5f}", (1, 0, 0, 0), (x - 0.02, y, z + 0.08))
        
        # distance_label.label_size = 0.1
        # distance_ray = self.viewer.viser_server.add_gui_modal("Distance")
        point3 = self.perspectiv.position + (point - self.perspectiv.position) * 0.001
        vertices = np.array([self.perspectiv.position, point, point3])
        faces = np.array([[0, 1, 2]])
        distance_ray = self.viewer.viser_server.add_mesh_simple(
            name="line_mesh",
            vertices=vertices,
            faces=faces,
            color=(155, 0, 0),
            wireframe=True, 
            opacity=None,
            material='standard',
            flat_shading=False,
            side='double',
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            visible=True
        )

        self.mesh_objs.append(distance_label)
        # self.mesh_objs.append(destity_label)
        self.mesh_objs.append(distance_ray)
        
        distance_ray.on_click(lambda _: distance_ray.remove())

        # with modal:
        #     self.viewer.viser_server.add_gui_markdown(f"Distance: {distance:.2f} m")
        #     # frame_origin = self.viewer.viser_server.add_frame("origin_frame", True, position=self.box.position, axes_length=0.3, axes_radius=0.01)
        #     # frame_target = self.viewer.viser_server.add_frame("frame_target", True, position=point, axes_length=0.3, axes_radius=0.01)
        #     # frame_origin.on_click(lambda _: frame_origin.remove())
        #     # frame_target.on_click(lambda _: frame_target.remove())
        #     self.viewer.viser_server.add_gui_button("Close").on_click(lambda _: modal.close())
    
    def toggle_labels(self):
        for obj in self.mesh_objs:
            if obj._impl.name == "density_label":
                obj.visible = not obj.visible
            if obj._impl.name == "distance_label":
                obj.visible = not obj.visible
                
    def compute_distance(self, a, b):
        """ 
        computes the distance between two points
        a: point a
        b: point b
        returns: distance
        """
        a = a.cpu().numpy() if isinstance(a, torch.Tensor) else a
        b = b.cpu().numpy() if isinstance(b, torch.Tensor) else b

        return (np.linalg.norm(np.array(a) - np.array(b)))
    
    def set_perspectiv_camera(self, type: str):
        
        clients = self.viewer.viser_server.get_clients()
        for id, client in clients.items():
            if type == "viser_perspectiv":
                client.camera.position = self.perspectiv.position
                client.camera.wxyz = self.perspectiv.wxyz
            else:
                self.perspectiv.position = client.camera.position
                self.perspectiv.wxyz = client.camera.wxyz
                x, y, z = self.perspectiv.position
                q_x, q_y, q_z = R.from_quat(self.perspectiv.wxyz).as_euler('xyz', degrees=True)
                self.perspectiv_pos_x.value = x
                self.perspectiv_pos_y.value = y
                self.perspectiv_pos_z.value = z
                self.perspectiv_wxyz_x.value = q_x
                self.perspectiv_wxyz_y.value = q_y
                self.perspectiv_wxyz_z.value = q_z

 
    def delete_point_cloud(self):
        if len(self.mesh_objs) > 0:
            for obj in self.mesh_objs:
                obj.remove()
            
    def check_interrupt(self, frame, event, arg):
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.interrupt_render_flag:
                self.interrupt_render_flag = False
                raise viewer_utils.IOChangeException
        return self.check_interrupt

    def _send_output_to_viewer(
        self, outputs: Dict[str, Any], static_render: bool = True
    ):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the model
        """
        output_keys = set(outputs.keys())
        if self.output_keys != output_keys:
            self.output_keys = output_keys
            self.viewer.control_panel.update_output_options(list(outputs.keys()))

        output_render = self.viewer.control_panel.output_render
        self.viewer.update_colormap_options(
            dimensions=outputs[output_render].shape[-1],
            dtype=outputs[output_render].dtype,
        )
        selected_output = colormaps.apply_colormap(
            image=outputs[self.viewer.control_panel.output_render],
            colormap_options=self.viewer.control_panel.colormap_options,
        )

        if self.viewer.control_panel.split:
            split_output_render = self.viewer.control_panel.split_output_render
            self.viewer.update_split_colormap_options(
                dimensions=outputs[split_output_render].shape[-1],
                dtype=outputs[split_output_render].dtype,
            )
            split_output = colormaps.apply_colormap(
                image=outputs[self.viewer.control_panel.split_output_render],
                colormap_options=self.viewer.control_panel.split_colormap_options,
            )
            split_index = min(
                int(
                    self.viewer.control_panel.split_percentage
                    * selected_output.shape[1]
                ),
                selected_output.shape[1] - 1,
            )
            selected_output = torch.cat(
                [selected_output[:, :split_index], split_output[:, split_index:]], dim=1
            )
            selected_output[:, split_index] = torch.tensor(
                [0.133, 0.157, 0.192], device=selected_output.device
            )

        selected_output = (selected_output * 255).type(torch.uint8)
        depth = (
            outputs["gl_z_buf_depth"].cpu().numpy() * self.viser_scale_ratio
            if "gl_z_buf_depth" in outputs
            else None
        )

        # Convert to numpy.
        selected_output = selected_output.cpu().numpy()
        assert selected_output.shape[-1] == 3

        # Pad image if the aspect ratio (W/H) doesn't match the client!
        current_h, current_w = selected_output.shape[:2]
        desired_aspect = self.client.camera.aspect
        pad_width = int(max(0, (desired_aspect * current_h - current_w) // 2))
        pad_height = int(max(0, (current_w / desired_aspect - current_h) // 2))
        if pad_width > 5 or pad_height > 5:
            selected_output = np.pad(
                selected_output,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        jpg_quality = (
            self.viewer.config.jpeg_quality
            if static_render
            else 75 if self.viewer.render_tab_state.preview_render else 40
        )
        self.client.set_background_image(
            selected_output,
            format=self.viewer.config.image_format,
            jpeg_quality=jpg_quality,
            depth=depth,
        )
        res = f"{selected_output.shape[1]}x{selected_output.shape[0]}px"
        self.viewer.stats_markdown.content = self.viewer.make_stats_markdown(None, res)

    def _calculate_image_res(self, aspect_ratio: float) -> Tuple[int, int]:
        """Calculate the maximum image height that can be rendered in the time budget

        Args:
            apect_ratio: the aspect ratio of the current view
        Returns:
            image_height: the maximum image height that can be rendered in the time budget
            image_width: the maximum image width that can be rendered in the time budget
        """
        max_res = self.viewer.control_panel.max_res
        if self.state == "high":
            # high res is always static
            image_height = max_res
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)
        elif self.state in ("low_move", "low_static"):
            if (
                writer.is_initialized()
                and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
            ):
                vis_rays_per_sec = GLOBAL_BUFFER["events"][
                    EventName.VIS_RAYS_PER_SEC.value
                ]["avg"]
            else:
                vis_rays_per_sec = 100000
            target_fps = self.target_fps
            num_vis_rays = vis_rays_per_sec / target_fps
            image_height = (num_vis_rays / aspect_ratio) ** 0.5
            image_height = int(round(image_height, -1))
            image_height = max(min(max_res, image_height), 30)
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)
        else:
            raise ValueError(f"Invalid state: {self.state}")

        return image_height, image_width


 # cumulative methode: -------------------------------------------------------------------------------------------
                # distance, location, density = self.find_collision(ray_locations, ray_densities, self.threshold_slider.value)
                # if distance is not None:
                #     filtered_locations.append(location.tolist()) #type: ignore
                #     filtered_densities.append(density.item()) #type: ignore
                    
                 # difference methode (current): ----------------------------------------------------------------------------------------------
                    
                # first_iteration = True
                # density_differece = 0
                # previous_density = 0
                # for location, density in zip(ray_locations, ray_densities):
                #     if first_iteration:
                #         first_iteration = False
                #         previous_density = density
                #         continue
                #     if not first_iteration:
                #         density_differece = abs(density - previous_density)
                #     if not first_iteration and density_differece > self.density_threshold:
                #         filtered_locations.append(location.tolist())
                #         filtered_densities.append(density)
                #         break
                    
            # standardized densities methode: -----------------------------------------------------------------------------------
            
                # global_mean = torch.mean(all_densities)
                # global_std = torch.std(all_densities)
                # standardized_densities = (ray_densities - global_mean) / global_std
            
                # mask = standardized_densities.squeeze() > self.density_threshold
                # if torch.any(mask):
                #     first_index = torch.where(mask)[0][0]
                #     filtered_locations.append(ray_locations[first_index].unsqueeze(0))
                # else:
                #     pass

            # density sum methode: ---------------------------------------------------------------------------------------------
            
                # density_sum = 0 
                # for location, density in zip(ray_locations, ray_densities):
                #     density_sum += density.item()  # sum of densities
                #     if density_sum >= self.density_threshold:
                #         filtered_locations.append(location.tolist())    
                #         break
                    
            # single point theshold methode: ------------------------------------------------------------------------------------------------
                
                # for location, density in zip(ray_locations, ray_densities):
                #     if density > self.density_threshold:
                #         filtered_locations.append(location.tolist())   
                #         filtered_densities.append(density) 
                #         break

                # largest density difference methode: ---------------------------------------------------------------------
                
                # density_diff_list = []
                # first_iteration = True
                # previous_density = 0

                # for location, density in zip(ray_locations, ray_densities):
                #     if first_iteration:
                #         first_iteration = False
                #         previous_density = density
                #     else:
                #         density_difference = abs(density - previous_density)
                #         density_diff_list.append(density_difference)
                #         previous_density = density

                # Debugging.log("density_diff_list:", density_diff_list)

                # Find the index of the maximum difference
                # if density_diff_list:
                #     max_index = 0
                #     max_element = density_diff_list[0]
                #     for i in range(1, len(density_diff_list)):
                #         if density_diff_list[i] > max_element:
                #             max_element = density_diff_list[i]
                #             max_index = i + 1  # +1 to align with ray_locations and ray_densities index
            
                # first largest density methode: ---------------------------------------------------------------------------
            
                # # Initialisierung der Variablen
                # density_diff_list = []
                # first_iteration = True
                # previous_density = 0

                #
                # for location, density in zip(ray_locations, ray_densities):
                #     if first_iteration:
                #         first_iteration = False
                #         previous_density = density
                #     else:
                #         density_difference = abs(density - previous_density)
                #         density_diff_list.append(density_difference)
                #         previous_density = density

                # # Debugging.log("density_diff_list:", density_diff_list)

                # # Berechnung der durchschnittlichen Differenz (ohne die erste, die Null ist)
                # if density_diff_list:
                #     avg_density_difference = sum(density_diff_list) / len(density_diff_list)
                #     Debugging.log("Average density difference:", avg_density_difference)

                #     # Finden des ersten signifikanten Anstiegs
                #     first_iteration = True
                #     previous_density = 0
                #     for i, (location, density) in enumerate(zip(ray_locations, ray_densities)):
                #         if first_iteration:
                #             first_iteration = False
                #             previous_density = density
                #         else:
                #             density_difference = abs(density - previous_density)
                #             if density_difference > avg_density_difference * 2:  # z.B. 2-fache der durchschnittlichen Differenz
                #                 max_index = i
                #                 break
                #             previous_density = density

                #     Debugging.log("Max density difference:", density_difference)
                #     Debugging.log("Max density index:", max_index)

                #     # Verwendung des Indexes, falls gefunden
                #     if max_index != -1:
                #         filtered_locations.append(ray_locations[max_index].tolist())
                #         filtered_densities.append(ray_densities[max_index])

                # filtered_locations.append(ray_locations[max_index-1].tolist())
                # filtered_densities.append(ray_densities[max_index-1])
                # Debugging.log("densitiy:", ray_densities[max_index-1])
            # show all densities methode:
                # for location, density in zip(ray_locations, ray_densities):
                #     if density > 0 and density < self.density_threshold:
                #         filtered_locations.append(location.tolist())
                #         filtered_densities.append(density)
                # print(ray_locations[index].tolist())
                
                
                # largest treshold density difference methode: -------------------------------------------------------------
                
                # density_diff_list = []             
                # first_iteration = True
                # previous_density = 0

                # for location, density in zip(ray_locations, ray_densities):
                #     if first_iteration:
                #         first_iteration = False
                #         previous_density = density
                #     else:
                #         density_difference = abs(density - previous_density)
                #         if density_difference >= self.density_threshold:
                #             filtered_locations.append(location.tolist())
                #             filtered_densities.append(density)
                #             break
                
                # zscore methode: ---------------------------------------------------------------------------------------
                
                # from scipy.stats import zscore

                    
                # ray_densities = np.array(ray_densities)
                # z_scores = zscore(ray_densities)

                # for i, z in enumerate(z_scores):
                #     if z > self.density_threshold:
                #         filtered_locations.append(ray_locations[i].tolist())
                #         filtered_densities.append(ray_densities[i])
                        # break       
                        
                # average density methode: ----------------------------------------------------------------------------------------
                
                # density_sum = 0
                # # first_iteration = True
                # # previous_density = 0
                # count = 0
                # for density in ray_densities:
                #     if density > 0:
                #         count += 1
                #         density_sum += density
                
                # average_density = density_sum / count
                
                # for location, density in zip(ray_locations, ray_densities):
                #     if density > (average_density-self.density_threshold):
                #         filtered_locations.append(location.tolist())
                #         filtered_densities.append(density)
                
                # average density difference methode: ----------------------------------------------------------------------------------------
                
                # density_difference_sum = 0
                # first_iteration = True
                # previous_density = 0
                # count = 0

                # for density in ray_densities:
                #     if first_iteration:
                #         first_iteration = False
                #         previous_density = density
                #     else:
                #         # if density > 0:
                #         density_difference_sum += abs(density - previous_density)
                #         count += 1
                #         previous_density = density
                
                # average_density_difference_sum = density_difference_sum / count
                # for location, density in zip(ray_locations, ray_densities):
                #     if density >= (average_density_difference_sum + self.density_threshold):
                #         filtered_locations.append(location.tolist())
                #         filtered_densities.append(density)
                #         break
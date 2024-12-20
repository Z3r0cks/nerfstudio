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
import csv
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, get_args

import numpy as np
import torch
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as R
from viser import ClientHandle

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils import colormaps, writer
#-------------------------------------------------------------
from nerfstudio.utils.debugging import Debugging
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.utils import CameraState, get_camera
from nerfstudio.viewer_legacy.server import viewer_utils

#-------------------------------------------------------------
VISER_NERFSTUDIO_SCALE_RATIO: float = 1.0

if TYPE_CHECKING:
    from nerfstudio.viewer.viewer import Viewer
    from nerfstudio.viewer.viewer_density import Viewer_density

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

    def __init__(self, viewer: Viewer_density, viser_scale_ratio: float, client: ClientHandle):
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
        self.mesaurement_point_coordinates: list[list[float] | str] = ["Not Set", "Not Set"]
        self.ray_id = 0
        self.side_id = 0
        self.density_threshold = 1e-5000
        self.fov_x = 1
        self.fov_y = 1
        self.width = 1
        self.height = 1
        self.mesh_objs = []
        self.dataparser_transforms = {'transform': [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]}
        self.compute_scale_factor = float(1)
        self.dataparser_transforms = {'transform': [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]}

        with open(self.viewer.dataparser_transforms_path) as f:
                dataparser_transforms = json.load(f)
        
        self.translate_pos_from_omnivers = (dataparser_transforms["transform"][0][3], dataparser_transforms["transform"][1][3], dataparser_transforms["transform"][2][3])
        self.x_omni, self.y_omni, self.z_omni = self.translate_pos_from_omnivers
        self.add_frustum_btn()

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
    
    def delete_measurement_point(self, id):
        if id == 1:
            self.mesaurement_point_coordinates[0] = "Not Set"
            self.m_point_1.__setattr__("content", "Not Set")
            
        elif id == 2:
            self.mesaurement_point_coordinates[1] = "Not Set"
            self.m_point_2.__setattr__("content", "Not Set")
            
        elif id == 3:
            self.compute_scale_factor = float(1)
            self.scale_factor_info.__setattr__("content", f'Scale factor for distance measurement: {str(self.compute_scale_factor)}')

    def _open_calibrate_modal(self):
        viser = self.viewer.viser_server 
        with viser.gui.add_modal("Modal example") as modal:

            point_coordinate_1 = str(self.mesaurement_point_coordinates[0])
            point_coordinate_2 = str(self.mesaurement_point_coordinates[1])
            viser.gui.add_markdown("**Calibrate Distance Measurement**")
            viser.gui.add_markdown("**Coordinate Measurement Point 1:**")
            viser.gui.add_markdown(f'Point 1: {point_coordinate_1}')
            viser.gui.add_markdown("**Coordinate Measurement Point 2:**")
            viser.gui.add_markdown(f'Point 2: {point_coordinate_2}')
            
            if self.mesaurement_point_coordinates[0] == "Not Set" or self.mesaurement_point_coordinates[1] == "Not Set":
                viser.gui.add_markdown("Please set both coordinates first with precise measurement")
            else:
                second_modal_button = viser.gui.add_button("Use this Coordinates")
                @second_modal_button.on_click
                
                def _(_) -> None:
                    l_scene = self.compute_distance(self.mesaurement_point_coordinates[0], self.mesaurement_point_coordinates[1])
                    with viser.gui.add_modal("Calibrate") as second_modal:
                        viser.gui.add_markdown(f'Calibrate distance: {l_scene}')
                        l_real = viser.gui.add_number("Real distance in meter:", 1.0, 0.0, None, 0.0001)
                        
                        calibrate_button = viser.gui.add_button("Calibrate", color="green")
                        viser.gui.add_button("Close", color="red").on_click(lambda _: second_modal.close())
                        @calibrate_button.on_click
                        def _(_) -> None:
                            self.compute_scale_factor = 1
                            self.compute_scale_factor = l_scene / l_real.value
                            self.scale_factor_info.__setattr__("content", f'Scale factor for distance measurement: {str(self.compute_scale_factor)}')
                            self.mesaurement_point_coordinates = ["Not Set", "Not Set"]
                            modal.close()
                            second_modal.close()
                            self.delete_point_cloud()
                            
            viser.gui.add_button("Close", color="red").on_click(lambda _: modal.close())       
    

    def add_frustum_btn(self):
        self.viewer.viser_server.gui.set_panel_label("LiDAR NeRF Studio")
        self.viewer.viser_server.gui.configure_theme(brand_color=(2, 125, 189), dark_mode=True, control_layout="floating")
        add_frustum_rgb = self.viewer.viser_server.gui.add_rgb("Color Of Frustum", (235, 52, 79))
        add_frustum_button = self.viewer.viser_server.gui.add_button("Add Frustum and GUI")
        
        @add_frustum_button.on_click
        def _(_) -> None:
            self.generate_lidar_gui(add_frustum_rgb.value)
            add_frustum_rgb.remove()
            add_frustum_button.__setattr__("visible", False)
        
    def generate_lidar_gui(self, rgb) -> None:
        '''Add GUI elements for LiDAR'''    
        
        viser = self.viewer.viser_server
        self.frustum = viser.scene.add_camera_frustum(name="camera_frustum", fov=5.0, aspect=1, scale=0.1, color=rgb, wxyz=(1, 0, 0, 0), position=(self.x_omni, self.y_omni, self.z_omni))
        self.scale_factor_info =  viser.gui.add_markdown(f'Scale factor for distance measurement: {str(self.compute_scale_factor)}')
        
        with viser.gui.add_folder("Point Cloud Settings", expand_by_default=False):
            self.point_cloud_color = self.viewer.viser_server.gui.add_rgb("Point Cloud Color", (255, 0, 224))
            self.point_cloud_base_size = self.viewer.viser_server.gui.add_number("Point Cloud Base Size", 0.007, 0.0005, 0.2, 0.0001)
            self.max_distance = viser.gui.add_slider("Max Distance (0 = Max)", 0, 1000, 0.00001, 0)
        
        with viser.gui.add_folder("Frustum Positioning", expand_by_default=False):
            with viser.gui.add_folder("Frustum Location"):
                self.frustum_pos_x = viser.gui.add_slider("Pos X", -20, 20, 0.001, 0)
                self.frustum_pos_y = viser.gui.add_slider("Pos Y", -20, 20, 0.001, 0)
                self.frustum_pos_z = viser.gui.add_slider("Pos Z", -20, 20, 0.001, 0)
        
            with viser.gui.add_folder("Frustum Orientation"):
                self.frustumv_wxyz_x = viser.gui.add_slider("Rot X", -180, 180, 0.1, 0)
                self.frustum_wxyz_y = viser.gui.add_slider("Rot Y", -180, 180, 0.1, 0)
                self.frustum_wxyz_z = viser.gui.add_slider("Rot Z", -180, 180, 0.1, 0)
        
        with open(os.path.join(Path(__file__).resolve().parents[2], 'lidar_settings.json')) as f:
            lidar_data = json.load(f)
            
        with viser.gui.add_folder("LiDAR Sensors", expand_by_default=False):
            with viser.gui.add_folder("Resolution"):
                self.h_angle_resolution_dropdown = viser.gui.add_dropdown("Horizontal Resolution", ["0.125", "0.25", "0.5", "1", "2", "3", "4", "5"], "1")
                self.v_angle_resolution_dropdown = viser.gui.add_dropdown("Vertical Angle Resolution", ["0.125", "0.25", "0.5", "1", "2", "3", "4", "5"], "1")
            
            for lidar in lidar_data:
                scanner_settings = lidar_data[lidar]
                with viser.gui.add_folder(lidar_data[lidar]["name"], expand_by_default=False):
                    viser.gui.add_markdown("Angle Resolution: " + str(lidar_data[lidar]["_angle_resolution"]))
                    viser.gui.add_button("Generate Point Cloud", color="blue").on_click(lambda _, scanner_settings=scanner_settings: self.generate_lidar(scanner_settings=scanner_settings))
                    viser.gui.add_button("Clickable Point Cloud", color="violet").on_click(lambda _, scanner_settings=scanner_settings: self.generate_lidar(clickable=True, scanner_settings=scanner_settings))
                    viser.gui.add_button("Generate Plot", color="cyan").on_click(lambda _, scanner_settings=scanner_settings: self.generate_lidar(plot_density=True, scanner_settings=scanner_settings))
                    viser.gui.add_button("Show All Rays", color="teal").on_click(lambda _, scanner_settings=scanner_settings: self.generate_lidar(debugging=True, scanner_settings=scanner_settings))
            
        with viser.gui.add_folder("Measurement", expand_by_default=False):
            with viser.gui.add_folder("Resolution Settings", expand_by_default=False):
                with viser.gui.add_folder("Width (X)"):
                    self.frustum_fov_x = viser.gui.add_slider("FOV Horizontal", 0, 360, 1, self.fov_x)
                    self.frustum_width = viser.gui.add_slider("Number Of Rays", 1, 10000, 1, self.width)
                    
                with viser.gui.add_folder("Height (Y)"):
                    self.frustum_fov_y = viser.gui.add_slider("FOV Vertical", 0, 360, 1, self.fov_y)
                    self.frustum_heigth = viser.gui.add_slider("Number Of Rays", 1, 10000, 1, self.height)
                    
            with viser.gui.add_folder("Precise Measurement", expand_by_default=False):
                viser.gui.add_button("Measure Point 1", color="violet").on_click(lambda _: self.generate_lidar(measure=[True, 0]))
                viser.gui.add_button("Measure Point 2", color="violet").on_click(lambda _: self.generate_lidar(measure=[True, 1]))
                viser.gui.add_markdown("Coord Measurement Point 1")
                self.m_point_1 = viser.gui.add_markdown("Not Set")
                viser.gui.add_markdown("Coord Measurement Point 2")
                self.m_point_2 = viser.gui.add_markdown("Not Set")
                viser.gui.add_button("Open Calibrate Modal", color="teal").on_click(lambda _: self._open_calibrate_modal())
                
            with viser.gui.add_folder("individual Measurement", expand_by_default=False):    
                viser.gui.add_button("Generate Point Cloud", color="blue").on_click(lambda _: self.generate_lidar())
                viser.gui.add_button("Clickable Point Cloud", color="violet").on_click(lambda _: self.generate_lidar(clickable=True))
                viser.gui.add_button("Generate Plot", color="cyan").on_click(lambda _: self.generate_lidar(plot_density=True))
                viser.gui.add_button("Show All Rays", color="teal").on_click(lambda _: self.generate_lidar(debugging=True))
                
            with viser.gui.add_folder("Editing", expand_by_default=False):
                viser.gui.add_button("Toggle Labels", color="cyan").on_click(lambda _: self.toggle_labels())
                viser.gui.add_button("Delete Point 1", color="cyan").on_click(lambda _: self.delete_measurement_point(1))
                viser.gui.add_button("Delete Point 2", color="cyan").on_click(lambda _: self.delete_measurement_point(2))
                viser.gui.add_button("Set Scale Factor To 1", color="violet").on_click(lambda _: self.delete_measurement_point(3))


        self.frustum_pos_x.on_update(lambda _: self.update_cube())
        self.frustum_pos_y.on_update(lambda _: self.update_cube())
        self.frustum_pos_z.on_update(lambda _: self.update_cube())
        self.frustumv_wxyz_x.on_update(lambda _: self.update_cube())
        self.frustum_wxyz_y.on_update(lambda _: self.update_cube())
        self.frustum_wxyz_z.on_update(lambda _: self.update_cube())

        
        self.frustum_fov_x.on_update(lambda _: setattr(self, "fov_x", self.frustum_fov_x.value))
        self.frustum_fov_y.on_update(lambda _: setattr(self, "fov_y", self.frustum_fov_y.value))
        self.frustum_heigth.on_update(lambda _: setattr(self, "height", self.frustum_heigth.value))
        self.frustum_width.on_update(lambda _: setattr(self, "width", self.frustum_width.value))
            
        viser.gui.add_button("Clear Point Cloud", color="red").on_click(lambda _: self.delete_point_cloud())

    def update_cube(self):
        self.frustum.wxyz = R.from_euler('xyz', [self.frustumv_wxyz_x.value, self.frustum_wxyz_y.value, self.frustum_wxyz_z.value], degrees=True).as_quat()
        self.frustum.position = (self.frustum_pos_x.value + self.x_omni, self.frustum_pos_y.value + self.y_omni, self.frustum_pos_z.value + self.z_omni)

    def cartograph_env(self) -> None:
        print_list = []
        
        for v in range(450):
            Rv = vtf.SO3(wxyz=self.frustum.wxyz)
            Rv = Rv @ vtf.SO3.from_x_radians(np.pi)
            Rv = torch.tensor(Rv.as_matrix())
            origin = torch.tensor(self.frustum.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
            c2w = torch.concatenate([Rv, origin[:, None]], dim=1)
            
            fov_x = self.fov_x
            fov_y = self.fov_y
            fx_value = self.width / (2 * math.tan(math.radians(fov_x / 2)))
            fy_value = self.height / (2 * math.tan(math.radians(fov_y / 2)))
            
            fx = torch.tensor([[fx_value]], device='cuda:0')
            fy = torch.tensor([[fy_value]], device='cuda:0')
            cx = torch.tensor([[self.width / 2]], device='cuda:0')
            cy = torch.tensor([[self.height / 2]], device='cuda:0')
            
            camera = Cameras(
                camera_to_worlds=c2w,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=torch.tensor([[self.width]]),
                height=torch.tensor([[self.height]]),
                distortion_params=None,
                camera_type=10,
                times=torch.tensor([[0.]], device='cuda:0')
            )
            
            assert isinstance(camera, Cameras)
            outputs = self.viewer.get_model().get_outputs_for_camera(camera, width=self.width, height=self.height)

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
            
            for ray_locations, ray_densities in zip(all_density_locations, all_densities):
                if ray_densities.numel() == 0:
                    continue
                
                distance, location, density = self.find_collision_with_transmittance(ray_locations, ray_densities)
                try:
                    n_location = location.cpu().numpy() #type: ignore
                except:
                    continue
                
                filtered_locations.append(n_location)
        
            for location in filtered_locations:
                print_list.append(location)
            
            x, y, z = self.frustum.position #type: ignore
            self.frustum.position = x + 0.01, y, z #type: ignore
            
        import matplotlib.cm as cm
        import open3d as o3d
            
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(print_list)

        origin_np = origin = (torch.tensor(self.frustum.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO).numpy()
        distances = np.linalg.norm(print_list - origin_np, axis=1)
        min_distance = distances.min()
        max_distance = distances.max()
        normalized_distances = (distances - min_distance) / (max_distance - min_distance)
        colormap = cm.get_cmap('viridis')
        colors = colormap(normalized_distances)[:, :3] #type: ignore

        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([point_cloud])  # type: ignore
        
        self.add_column_to_csv(print_list)
          
    def _scan_density(self) -> None:
        print_list = []
        
        for v in range(50):
            Rv = vtf.SO3(wxyz=self.frustum.wxyz)
            Rv = Rv @ vtf.SO3.from_x_radians(np.pi)
            Rv = torch.tensor(Rv.as_matrix())
            origin = torch.tensor(self.frustum.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
            c2w = torch.concatenate([Rv, origin[:, None]], dim=1)
            
            fov_x = self.fov_x
            fov_y = self.fov_y
            fx_value = self.width / (2 * math.tan(math.radians(fov_x / 2)))
            fy_value = self.height / (2 * math.tan(math.radians(fov_y / 2)))
            
            fx = torch.tensor([[fx_value]], device='cuda:0')
            fy = torch.tensor([[fy_value]], device='cuda:0')
            cx = torch.tensor([[self.width / 2]], device='cuda:0')
            cy = torch.tensor([[self.height / 2]], device='cuda:0')
            
            camera = Cameras(
                camera_to_worlds=c2w,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=torch.tensor([[self.width]]),
                height=torch.tensor([[self.height]]),
                distortion_params=None,
                camera_type=10,
                times=torch.tensor([[0.]], device='cuda:0')
            )
            
            assert isinstance(camera, Cameras)
            outputs = self.viewer.get_model().get_outputs_for_camera(camera, width=self.width, height=self.height)

            all_densities = []
            all_density_locations = []
            
            for densities, locations in zip(outputs["densities"], outputs["densities_locations"]):
                if densities.numel() > 0:
                    all_densities.append(densities)
                if locations.numel() > 0:
                    all_density_locations.append(locations)
                    
            all_densities = torch.cat(all_densities)
            all_density_locations = torch.cat(all_density_locations)

            filtered_distances = []
            
            for ray_locations, ray_densities in zip(all_density_locations, all_densities):
                if ray_densities.numel() == 0:
                    continue
                
                distance, location, density = self.find_collision_with_transmittance(ray_locations, ray_densities)
                distance = self.compute_distance(self.frustum.position, location)
                    
                filtered_distances.append(distance)
        
            for distance in filtered_distances:
                print_list.append(distance)
                
            self.add_column_to_csv(print_list)
            print_list.clear()
            x, y, z = self.frustum.position #type: ignore
            self.frustum.position = x + 0.05, y, z #type: ignore
     
    def add_column_to_csv(self, new_column_data):
        self.csv_filename = 'cato.csv'
        try:
            with open(self.csv_filename, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                data = list(reader)
        except FileNotFoundError:
            data = []
            
        if not data:
            for value in new_column_data:
                data.append([str(value)])
        else:
            for i in range(len(new_column_data)):
                if i < len(data):
                    data[i].append(str(new_column_data[i]))
                else:
                    row = [''] * (len(data[0]) - 1) + [str(new_column_data[i])]
                    data.append(row)

        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerows(data)       
            
        
    def generate_lidar(self, **kwargs) -> None:
        """Show the density in the viewer

        kwargs:
            plot_density: bool = False -> if the density should be plotted
            clickable: bool = False -> if every point should be clickable (very slow)
            showNearesDensity: bool = False -> show the nearesDenstiy to 1
            showSingelRayInf: bool = False -> Save the information of a single ray
            debugging: bool = False -> Show all points of the density
            scanner_settings: None | str = None -> the settings of the lidar scanner
            measuer: list[bool, int] = [False, 0] -> use precise measurement
            point_cloud_color: tuple[int, int, int] = (255, 0, 224) -> the color of the point cloud
            point_cloud_base_size: float = 0.05 -> the base size of the point cloud
        """
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
        
        Rv = vtf.SO3(wxyz=self.frustum.wxyz)
        Rv = Rv @ vtf.SO3.from_x_radians(np.pi)
        Rv = torch.tensor(Rv.as_matrix())
        origin = torch.tensor(self.frustum.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
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
            
            if self.max_distance.value > 0:
                ray_locations, ray_densities = self.filter_max_distance(ray_locations, ray_densities)
            
            # takte the point which is nearest to nearestDistanceToCamera
            if showNearesDensity:
                distance_search = 1
                min_difference = float('inf') 
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
            import matplotlib.cm as cm
            import open3d as o3d
            
               # 3D Punktwolke erstellen
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(filtered_locations)

            origin_np = origin.numpy()
            distances = np.linalg.norm(filtered_locations - origin_np, axis=1)
            min_distance = distances.min()
            max_distance = distances.max()
            normalized_distances = (distances - min_distance) / (max_distance - min_distance)
            colormap = cm.get_cmap('viridis')
            colors = colormap(normalized_distances)[:, :3] #type: ignore

            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            o3d.visualization.draw_geometries([point_cloud])  # type: ignore
        
        if not plot_density and clickable == None and not showNearesDensity and not showSingelRayInf:
            if len(self.mesh_objs) > 0:
                self.delete_point_cloud()
            
            Debugging.log("filtered_locations", filtered_locations.shape)
            if filtered_locations.shape != (0,):
                obj = self.viewer.viser_server.scene.add_point_cloud(name="density", points=filtered_locations*VISER_NERFSTUDIO_SCALE_RATIO, colors=self.point_cloud_color.value, point_size=self.point_cloud_base_size.value, wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0), visible=True)
                self.mesh_objs.append(obj)
            else: 
                print("no data")
        
        if clickable != None or measure[0] != False:
            if len(self.mesh_objs) > 0:
                self.delete_point_cloud()
                
            for index, (location, density) in enumerate(zip(filtered_locations, filtered_densities)):
                self.add_point_as_mesh(location, index, density, measure=measure)
                
    def filter_max_distance(self, ray_locations, ray_densities):
        filtered_locations = []
        filtered_densities = []

        for location, density in zip(ray_locations[1:], ray_densities[1:]):
            distance = self.compute_distance(self.frustum.position, location)
            if distance <= self.max_distance.value:
                filtered_locations.append(location)
                filtered_densities.append(density)

        return filtered_locations, filtered_densities
        
    # concept of transmittance T(t), describes the probability of a photon to pass through a medium without being absorbed
    # If the transmittance is low, the photon is more likely to be absorbed. Start with transmittance = 1 and multiply it with the transmittance of each point along the ray.
    def find_collision_with_transmittance(self, ray_locations, ray_densities, transmission_threshold=1e-200):
        """
        Finds the collision point along a ray based on transmission values.
        
        ray_locations: Tensor of 3D coordinates representing points along the ray.
        ray_densities: Tensor of density values corresponding to each point.
        transmission_threshold: The threshold for the transmission probability to consider as a collision.
        
        returns: Distance from the origin to the collision point, the collision location and density.
        """
        transmittance = 1.0  # Initial transmittance
        origin = ray_locations[0]
        
        for location, density in zip(ray_locations[1:], ray_densities[1:]):
            distance = self.compute_distance(origin, location)
                
            delta_transmittance = torch.exp(-density * distance)
            transmittance *= delta_transmittance
            
            if transmittance < transmission_threshold:
                return distance, location, density

        return None, None, None  # No collision found
        
    def add_point_as_mesh(self, location, index, density, scale_factor=1, measure=[False, 0]):
        
        base_size = self.point_cloud_base_size.value + 0.002
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
        obj = self.viewer.viser_server.scene.add_mesh_simple(
            name=mesh_name,
            vertices=vertices,
            faces=faces,
            color=self.point_cloud_color.value,
            position=(0, 0, 0),
            visible=True,
        )
        
        if measure[0]:
            obj.on_click(lambda _: self.add_distance_modal_between_points(location*VISER_NERFSTUDIO_SCALE_RATIO, obj, measure[1]))
        else:
            obj.on_click(lambda _: self.add_distance_modal(location*VISER_NERFSTUDIO_SCALE_RATIO, density))
        
        self.mesh_objs.append(obj)
        
    def print_nearest_density(self, distance, density):
        position = self.frustum.position.tolist() if isinstance(self.frustum.position, np.ndarray) else self.frustum.position
        rot = [self.frustum_pos_x.value, self.frustum_pos_y.value, self.frustum_pos_z.value]
        
        distance = float(distance) if isinstance(distance, float) else distance
        density = density.item() if isinstance(density, torch.Tensor) else density
        
        self.csv_filename = 'validation.csv'
        try:
            with open(self.csv_filename, 'x', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                headers = ["id", "position_x", "position_y", "position_z", "rotation_x", "rotation_y", "rotation_z", "distance", "density"]
                csvwriter.writerow(headers)
        except FileExistsError:
            pass

        with open(self.csv_filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            row = [self.ray_id] + position + rot + [distance, density]
            csvwriter.writerow(row)
            
    def add_distance_modal_between_points(self, coordinates, obj, point_id):
        point_button = self.m_point_2 if point_id == 1 else self.m_point_1
        
        self.mesaurement_point_coordinates[point_id] = coordinates
        self.set_measurement_point(point_button, coordinates, obj)
        
        
        if not isinstance(self.mesaurement_point_coordinates[0], str) and not isinstance(self.mesaurement_point_coordinates[1], str):
            self.mesaurement_point_coordinates[point_id] = coordinates
            (x1, y1, z1),(x2, y2, z2) = self.mesaurement_point_coordinates[0], self.mesaurement_point_coordinates[1]
            distance = self.compute_distance(self.mesaurement_point_coordinates[0], self.mesaurement_point_coordinates[1])
            
            distance_label = self.viewer.viser_server.scene.add_label("distance_label", f"Distance: {distance:.4f}", (1, 0, 0, 0), ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2))
            vertices = np.array([self.mesaurement_point_coordinates[0], self.mesaurement_point_coordinates[1], self.mesaurement_point_coordinates[0]])
            faces = np.array([[0, 1, 2]])
            
            distance_ray = self.viewer.viser_server.scene.add_mesh_simple(
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
            self.set_measurement_point(point_button, coordinates, obj)
            
    def set_measurement_point(self, point_button, coodinate, obj):
        point_button.__setattr__("content", f"{str(coodinate)}")
        obj.__setattr__("color", "red")
        
    def add_distance_modal(self, point, density):
        """ 
        add a modal to show the distance of a point
        point: point
        """
        x, y, z = point
        global global_distance
        global global_density
        global_distance = self.compute_distance(self.frustum.position, point)
        global_density = density
        distance_label = self.viewer.viser_server.scene.add_label("distance_label", f"Distance: {global_distance:.4f}", (1, 0, 0, 0), (x - 0.02, y, z + 0.04))

        point3 = self.frustum.position + (point - self.frustum.position) * 0.001
        vertices = np.array([self.frustum.position, point, point3])
        faces = np.array([[0, 1, 2]])
        distance_ray = self.viewer.viser_server.scene.add_mesh_simple(
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

        try:
            return (np.linalg.norm(np.array(a) - np.array(b))) / self.compute_scale_factor
        except:
            return -1
            
        
    
    def set_perspectiv_camera(self, type: str):
        
        clients = self.viewer.viser_server.get_clients()
        for id, client in clients.items():
            if type == "viser_perspectiv":
                client.camera.position = self.frustum.position
                client.camera.wxyz = self.frustum.wxyz
            else:
                self.frustum.position = client.camera.position
                self.frustum.wxyz = client.camera.wxyz
                x, y, z = self.frustum.position
                q_x, q_y, q_z = R.from_quat(self.frustum.wxyz).as_euler('xyz', degrees=True)
                self.frustum_pos_x.value = x
                self.frustum_pos_y.value = y
                self.frustum_pos_z.value = z
                self.frustumv_wxyz_x.value = q_x
                self.frustum_wxyz_y.value = q_y
                self.frustum_wxyz_z.value = q_z

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
        self.client.scene.set_background_image(
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

    # some functions I used for testing:

    # function to create screenshots for testing
    # def take_screenshot(self):
    #     import pyautogui
    #     # from PIL import Image
        
    #     file_name = "id_" + str(self.ray_id) + ".png"
    #     file_dir = "##"
    #     screenshot = pyautogui.screenshot()
        
    #     crop_size = 300  # 50 pixels in all directions means a total of 100x100 pixels

    #     # Get the dimensions of the screenshot
    #     width, height = screenshot.size

    #     # Calculate the cropping box centered in the screenshot
    #     left = (width - crop_size) // 2
    #     top = (height - crop_size) // 2
    #     right = left + crop_size
    #     bottom = top + crop_size
        
    #     cropped_image = screenshot.crop((left, top, right, bottom))
    #     cropped_image.save(file_dir + file_name)

    # def get_ray_infos(self) -> None:
    #     for i in range(50):
    #         self.generate_lidar(showNearesDensity=True)
    #     self.increment_ray_id()
    
    # def increment_ray_id(self) -> None:
    #     self.ray_id += 1
    #     Debugging.log("ray_id: ", self.ray_id)
    
    
    
     # def print_single_ray_informations(self, print_list):
    #     self.csv_filename = 'floor.csv'
    #     with open(self.csv_filename, 'a', newline='') as csvfile:
    #         csvwriter = csv.writer(csvfile, delimiter=';')
    #         csvwriter.writerow(print_list[0])
    # def _scan_density(self) -> None:
    #         print_list = []
    #         side_distance = 0
    #         for side in range(4):
    #             if side == 0:
    #                 side_distance = 1
    #                 self.frustum.position = side_distance + self.x_omni, 1.1 + self.y_omni, 1.9 + self.z_omni
    #             elif side == 1:
    #                 self.frustum.position = side_distance + self.x_omni, 1.1 + self.y_omni, 0.9 + self.z_omni
    #             elif side == 2:
    #                 side_distance = 4
    #                 self.frustum.position = side_distance + self.x_omni, + -5.9 + self.y_omni, 1.9 + self.z_omni
    #             else:
    #                 self.frustum.position = side_distance + self.x_omni, + -5.9 + self.y_omni, 0.9 + self.z_omni
    #             for horizont in range(8):
    #                 for vertical in range(8):
    #                     x, y, z = self.frustum.position #type: ignore
    #                     self.frustum.position = side_distance, y, z
                            
    #                     # for deep in range(8):
    #                     x, y, z = self.frustum.position #type: ignore
    #                     real_distance = 0

    #                     Rv = vtf.SO3(wxyz=self.frustum.wxyz)
    #                     Rv = Rv @ vtf.SO3.from_x_radians(np.pi)
    #                     Rv = torch.tensor(Rv.as_matrix())
    #                     origin = torch.tensor(self.frustum.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
    #                     c2w = torch.concatenate([Rv, origin[:, None]], dim=1)
                        
    #                     fov_x = self.fov_x
    #                     fov_y = self.fov_y
    #                     fx_value = self.width / (2 * math.tan(math.radians(fov_x / 2)))
    #                     fy_value = self.height / (2 * math.tan(math.radians(fov_y / 2)))
                        
    #                     fx = torch.tensor([[fx_value]], device='cuda:0')
    #                     fy = torch.tensor([[fy_value]], device='cuda:0')
    #                     cx = torch.tensor([[self.width / 2]], device='cuda:0')
    #                     cy = torch.tensor([[self.height / 2]], device='cuda:0')
                        
    #                     camera = Cameras(
    #                         camera_to_worlds=c2w,
    #                         fx=fx,
    #                         fy=fy,
    #                         cx=cx,
    #                         cy=cy,
    #                         width=torch.tensor([[self.width]]),
    #                         height=torch.tensor([[self.height]]),
    #                         distortion_params=None,
    #                         camera_type=10,
    #                         times=torch.tensor([[0.]], device='cuda:0')
    #                     )
                        
    #                     assert isinstance(camera, Cameras)
    #                     outputs = self.viewer.get_model().get_outputs_for_camera(camera, width=self.width, height=self.height)

    #                     all_densities = []
    #                     all_density_locations = []
                        
    #                     for densities, locations in zip(outputs["densities"], outputs["densities_locations"]):
    #                         if densities.numel() > 0:
    #                             all_densities.append(densities)
    #                         if locations.numel() > 0:
    #                             all_density_locations.append(locations)
                                
    #                     all_densities = torch.cat(all_densities)
    #                     all_density_locations = torch.cat(all_density_locations)
                
    #                     filtered_distances = []
    #                     filtered_densities = []
    #                     filtered_diff = []
                        
    #                     for ray_locations, ray_densities in zip(all_density_locations, all_densities):
    #                         if ray_densities.numel() == 0:
    #                             continue
                            
    #                         distance, location, density = self.find_collision_with_transmittance(ray_locations, ray_densities)
    #                         distance = self.compute_distance(self.frustum.position, location)
    #                         real_distance = round(distance - 1)
    #                         diff = real_distance - (distance - 1)
    #                         if type(density) != None:
    #                             density = density.cpu().numpy() #type: ignore
                                
    #                         filtered_distances.append(distance)
    #                         filtered_densities.append(float(density)) #type: ignore
    #                         filtered_diff.append(diff) #type: ignore
                        
    #                     for distance, density, diff in zip(filtered_distances, filtered_densities, filtered_diff):
    #                         print_list.append([real_distance, distance-1, diff, density])
    #                         self.print_single_ray_informations(print_list)
    #                         print_list.clear()
    #                         self.ray_id += 1
                                
    #                         self.frustum.position = x + 0.4, y, z #type: ignore
    #                         x, y, z = self.frustum.position #type: ignore
    #                     self.frustum.position = x, y + 0.1, z #type: ignore
    #                 x, y, z = self.frustum.position #type: ignore
    #                 self.frustum.position = x, y - 0.8, z - 0.1
    #             self.side_id += 1
                
                
    # def print_single_ray_informations(self, print_list):
    #     self.csv_filename = 'validation.csv'
    #         # "side id" "id", "location", "distance", "density", rgb
    #     try:
    #         with open(self.csv_filename, 'x', newline='') as csvfile:
    #             csvwriter = csv.writer(csvfile)
    #             headers = ["side_id", "ray_id", "real_distance", "distance", "diff", "density"]
    #             csvwriter.writerow(headers)
    #     except FileExistsError:
    #         pass
        
    #     for i, info in enumerate(print_list):
    #         with open(self.csv_filename, 'a', newline='') as csvfile:
    #             csvwriter = csv.writer(csvfile)
    #             row = [self.side_id] + [self.ray_id] + info
    #             csvwriter.writerow(row)
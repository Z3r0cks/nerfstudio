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
import torch.nn.functional as F
from viser import ClientHandle

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components.renderers import background_color_override_context
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils import colormaps, writer
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.utils import CameraState, get_camera
from nerfstudio.viewer_legacy.server import viewer_utils

#-------------------------------------------------------------
from nerfstudio.utils.debugging import Debugging
# from scipy.spatial import cKDTree
# from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation as R

#-------------------------------------------------------------

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

    def __init__(self, viewer: Viewer | ViewerDensity, viser_scale_ratio: float, client: ClientHandle):
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
        self.density_threshold = 0
        self.FOV = 60
        self.FOV_width = 512
        self.FOV_height = 275
        self.mesh_objs = []
        self.viewer.viser_server.add_gui_button("Add GUI").on_click(lambda _: self.add_gui())
        
        self.densities = []
        self.density_locations = []
    
    def void_id(self):
        self.densities = []
        self.density_locations = []
        
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
                # if isinstance(self.viewer.get_model(), SplatfactoModel):
                #     color = self.viewer.control_panel.background_color
                #     background_color = torch.tensor(
                #         [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0],
                #         device=self.viewer.get_model().device,
                #     )
                #     self.viewer.get_model().set_background(background_color)
                self.viewer.get_model().eval()
                step = self.viewer.step
                try:
                    # if self.viewer.control_panel.crop_viewport:
                    #     color = self.viewer.control_panel.background_color
                    #     if color is None:
                    #         background_color = torch.tensor([0.0, 0.0, 0.0], device=self.viewer.pipeline.model.device)
                    #     else:
                    #         background_color = torch.tensor(
                    #             [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0],
                    #             device=self.viewer.get_model().device,
                    #         )
                    #     with background_color_override_context(
                    #         background_color
                    #     ), torch.no_grad(), viewer_utils.SetTrace(self.check_interrupt):
                    #         outputs = self.viewer.get_model().get_outputs_for_camera(camera, obb_box=obb)
                    # else:
                    with torch.no_grad(), viewer_utils.SetTrace(self.check_interrupt):
                        outputs = self.viewer.get_model().get_outputs_for_camera(camera, obb_box=obb)
                except viewer_utils.IOChangeException:
                    self.viewer.get_model().train()
                    raise
                self.viewer.get_model().train()
            num_rays = (camera.height * camera.width).item()
            if self.viewer.control_panel.layer_depth:
             
                # if isinstance(self.viewer.get_model(), SplatfactoModel):
                #     # Gaussians render much faster than we can send depth images, so we do some downsampling.
                #     assert len(outputs["depth"].shape) == 3
                #     assert outputs["depth"].shape[-1] == 1

                #     desired_depth_pixels = {"low_move": 128, "low_static": 128, "high": 512}[self.state] ** 2
                #     current_depth_pixels = outputs["depth"].shape[0] * outputs["depth"].shape[1]
                #     scale = min(desired_depth_pixels / current_depth_pixels, 1.0)

                #     outputs["gl_z_buf_depth"] = F.interpolate(
                #         outputs["depth"].squeeze(dim=-1)[None, None, ...],
                #         size=(int(outputs["depth"].shape[0] * scale), int(outputs["depth"].shape[1] * scale)),
                #         mode="bilinear",
                #     )[0, 0, :, :, None]
                # else:
                # Convert to z_depth if depth compositing is enabled.
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
            
            self.densities.extend(outputs["density"])
            self.density_locations.extend(outputs["density_locations"])
            
            print("Density Stack Length: ", len(self.densities))

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
    
    def add_gui(self) -> None:
        
        with self.viewer.viser_server.add_gui_folder("Density Options FOV"):
            self.viewer.viser_server.add_gui_button("Create Densites FOV", color="green").on_click(lambda _: self._show_density(FOV=True))
            self.viewer.viser_server.add_gui_button("Clear FOV Stack", color="red").on_click(lambda _: self.void_id())
            self.viewer.viser_server.add_gui_button("Plot Densites", color="indigo").on_click(lambda _: self._show_density(True, True))
            self.viewer.viser_server.add_gui_button("FOV Coords", color="violet").on_click(lambda _: self.get_camera_coods())
            
        with self.viewer.viser_server.add_gui_folder("Density Options Box"):
            self.viewer.viser_server.add_gui_button("Create Density", color="green").on_click(lambda _: self._show_density())
            self.viewer.viser_server.add_gui_button("Plot Densites", color="indigo").on_click(lambda _: self._show_density(True))
            
        with self.viewer.viser_server.add_gui_folder("Box Settings"):
            self.box_fov = self.viewer.viser_server.add_gui_slider("Box FOV", 0, 360, 1, 60)
            self.box_heigth = self.viewer.viser_server.add_gui_slider("Box Height", 30, 1080, 1, 512)
            self.box_width = self.viewer.viser_server.add_gui_slider("Box Width", 30, 1920, 1, 275)

        with self.viewer.viser_server.add_gui_folder("Point Cloud Options"):
            self.viewer.viser_server.add_gui_button("Clear Point Cloud", color="red").on_click(lambda _: self.delete_point_cloud())
            
        self.box = self.viewer.viser_server.add_box("box", (43, 42, 65), (.2, .1, .1), (1, 0, 0, 0), (1.50, 0.5, -3.65))
        self.box = self.viewer.viser_server.add_box("box", (43, 42, 65), (.2, .1, .1), (1, 0, 0, 0), (1.50, 0.5, -3.65))
        
        with self.viewer.viser_server.add_gui_folder("Density Threshold"):
            self.threshold_slider = self.viewer.viser_server.add_gui_slider("Threshold", -5, 5, 0.1, 0)
            
        with self.viewer.viser_server.add_gui_folder("Box Position"):
            self.box_pos_x = self.viewer.viser_server.add_gui_slider("Pos X", -4, 4, 0.01, 1.50)
            self.box_pos_y = self.viewer.viser_server.add_gui_slider("Pos Y", -4, 4, 0.01, 0.5)
            self.box_pos_z = self.viewer.viser_server.add_gui_slider("Pos Z", -4, 4, 0.01, -3.65)
        
        with self.viewer.viser_server.add_gui_folder("Box WXYZ"):
            self.box_wxyz_x = self.viewer.viser_server.add_gui_slider("X", -180, 180, 0.1, 0)
            self.box_wxyz_y = self.viewer.viser_server.add_gui_slider("Y", -180, 180, 0.1, 0)
            self.box_wxyz_z = self.viewer.viser_server.add_gui_slider("Z", -180, 180, 0.1, 0)
            
        self.box_pos_x.on_update(lambda _: setattr(self.box, 'position', (self.box_pos_x.value, self.box_pos_y.value, self.box_pos_z.value)))
        self.box_pos_y.on_update(lambda _: setattr(self.box, 'position', (self.box_pos_x.value, self.box_pos_y.value, self.box_pos_z.value)))
        self.box_pos_z.on_update(lambda _: setattr(self.box, 'position', (self.box_pos_x.value, self.box_pos_y.value, self.box_pos_z.value)))

        self.threshold_slider.on_update(lambda _: setattr(self, "density_threshold", self.threshold_slider.value))
        self.box_fov.on_update(lambda _: setattr(self, "FOV", self.box_fov.value))
        self.box_heigth.on_update(lambda _: setattr(self, "FOV_height", self.box_heigth.value))
        self.box_width.on_update(lambda _: setattr(self, "FOV_width", self.box_width.value))
        
        self.box_wxyz_x.on_update(lambda _: self.from_eul_to_quad())
        self.box_wxyz_y.on_update(lambda _: self.from_eul_to_quad())
        self.box_wxyz_z.on_update(lambda _: self.from_eul_to_quad())
        
        # self.viewer.viser_server.add_gui_button("Render in Viser", color="pink").on_click(lambda _: self.get_density(self.box.position))
    def from_eul_to_quad(self):
        self.box.wxyz = R.from_euler('xyz', [self.box_wxyz_x.value, self.box_wxyz_y.value, self.box_wxyz_z.value], degrees=True).as_quat()
        
    def _show_density(self, plot_density: bool = False, FOV: bool = False):
        """Show the density in the viewer

        Args:
            density_location: the density location
        """
        
                        
        # Cameras(camera_to_worlds=torch.tensor([[[ 0.2509, -0.9680, -0.0053,  0.1427],
        #     [ 0.9680,  0.2508,  0.0090,  0.0359],
        #     [-0.0074, -0.0074,  0.9999, -0.3990]]], device='cuda:0'), fx=torch.tensor([[26.0645]], device='cuda:0'), fy=torch.tensor([[26.0645]], device='cuda:0'), cx=torch.tensor([[37.]], device='cuda:0'), cy=torch.tensor([[20.]], device='cuda:0'), width=torch.tensor([[74]], device='cuda:0'), height=torch.tensor([[40]], device='cuda:0'), distortion_params=None, camera_type=torch.tensor([[1]], device='cuda:0'), times=torch.tensor([[0.]], device='cuda:0'), metadata=None)

        # Cameras(camera_to_worlds=torch.tensor([[[ 0.2509, -0.9680, -0.0053,  0.1427],
        #  [ 0.9680,  0.2508,  0.0090,  0.0359],
        #  [-0.0074, -0.0074,  0.9999, -0.3990]]], device='cuda:0'), fx=torch.tensor([[179.1935]], device='cuda:0'), fy=torch.tensor([[179.1935]], device='cuda:0'), cx=torch.tensor([[256.]], device='cuda:0'), cy=torch.tensor([[137.5000]], device='cuda:0'), width=torch.tensor([[512]], device='cuda:0'), height=torch.tensor([[275]], device='cuda:0'), distortion_params=None, camera_type=torch.tensor([[1]], device='cuda:0'), times=torch.tensor([[0.]], device='cuda:0'), metadata=None)

        # Cameras(camera_to_worlds=torch.tensor([[[ 0.2509, -0.9680, -0.0053,  0.1427],
        #     [ 0.9680,  0.2508,  0.0090,  0.0359],
        #     [-0.0074, -0.0074,  0.9999, -0.3990]]], device='cuda:0'), fx=torch.tensor([[703.7417]], device='cuda:0'), fy=torch.tensor([[703.7417]], device='cuda:0'), cx=torch.tensor([[960.]], device='cuda:0'), cy=torch.tensor([[540.]], device='cuda:0'), width=torch.tensor([[512]], device='cuda:0'), height=torch.tensor([[275]], device='cuda:0'), distortion_params=None, camera_type=torch.tensor([[1]], device='cuda:0'), times=torch.tensor([[0.]], device='cuda:0'), metadata=None)
        
        Debugging.log("Type", ("--------FOV--------" if FOV else "--------box--------"))
        
        if FOV:
            densities = self.densities
            density_locations = self.density_locations
        else:
            box_position = self.box.position    
            euler_angles = [self.box_wxyz_x.value, self.box_wxyz_y.value, self.box_wxyz_z.value]  # Euler-Winkel der Box

            # Konvertiere Euler-Winkel in eine Rotationsmatrix
            rotation_matrix = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
            rotation_matrix = torch.tensor(rotation_matrix, device='cuda:0', dtype=torch.float64)
                
            # camera_to_worlds = torch.tensor([
            #     [rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], box_position[0]],
            #     [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], box_position[1]],
            #     [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], box_position[2]]
            # ], device='cuda:0', dtype=torch.float32)
            camera_to_worlds = torch.tensor([
                [ 0.2509, -0.9680, -0.0053,  0.1427],
                [ 0.9680,  0.2508,  0.0090,  0.0359],
                [-0.0074, -0.0074,  0.9999, -0.3990]
            ], device='cuda:0', dtype=torch.float32)

            import math
            fx_value = self.FOV_width / (2 * math.tan(math.radians(self.FOV / 2)))
            fy_value = self.FOV_height / (2 * math.tan(math.radians(self.FOV / 2)))
            
            print("FOV:", self.FOV)
            print("FOV_height:", self.FOV_height)
            print("FOV_width:", self.FOV_width)
            print("fx:", fx_value)
            print("fy:", fy_value)
            fx = torch.tensor([[fx_value]], device='cuda:0')
            fy = torch.tensor([[fy_value]], device='cuda:0')
            # 512/2
            cx = torch.tensor([[self.FOV_width/2]], device='cuda:0')
            # 275/2
            cy = torch.tensor([[self.FOV_height/2]], device='cuda:0')

            camera = Cameras(
                camera_to_worlds=camera_to_worlds,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=torch.tensor([[self.FOV_width]]),
                height=torch.tensor([[self.FOV_height]]),
                distortion_params=None,
                camera_type=torch.tensor([[1]], device='cuda:0'),
                times=torch.tensor([[0.]], device='cuda:0')
            )

            assert isinstance(camera, Cameras)
            outputs = self.viewer.get_model().get_outputs_for_camera(camera, obb_box=None)

            # Extrahiere die Dichtewerte und Dichtepositionen
            densities = outputs["density"]
            density_locations = outputs["density_locations"]
        
        filtered_locations = []
        # filtered_densities = []
        # last_ray_point = []

        # Berechne globalen Mittelwert und Standardabweichung über alle Dichtewerte
        all_densities = torch.cat([ray_densities for ray_densities in densities if ray_densities.numel() > 0])

        
        global_mean = torch.mean(all_densities)
        global_std = torch.std(all_densities)

        for ray_locations, ray_densities in zip(density_locations, densities):
            if ray_densities.numel() == 0:
                continue

            # standardisieren
            standardized_densities = (ray_densities - global_mean) / global_std
            mask = standardized_densities.squeeze() > self.density_threshold
            # min_value = standardized_densities.min()
            if torch.any(mask):
                first_index = torch.where(mask)[0][0]
                filtered_locations.append(ray_locations[first_index].unsqueeze(0))
                # filtered_densities.append(ray_densities[first_index].unsqueeze(0))
            else:
                pass
            
        filtered_locations = torch.cat(filtered_locations)
        # # filtered_densities = torch.cat(filtered_densities)
        
        filtered_locations = filtered_locations.cpu().numpy()
        # filtered_densities = filtered_densities.cpu().numpy()

        remaining_indices = self.filter_nearby_indices(filtered_locations)
        filtered_locations = filtered_locations[remaining_indices]

        Debugging.log("filtered_locations", len(filtered_locations))
        
        if plot_density:
            import open3d as o3d
            # 3D point cloud visualisieren
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(filtered_locations)
            colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.ViewControl()
            o3d.visualization.draw_geometries([point_cloud])
        else:
            for index, location in enumerate(filtered_locations):
                self.add_point_as_mesh(location, index)

    def add_point_as_mesh(self, location, index, scale_factor=10, base_size=0.005, color=(255, 0, 255)):
            
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
        self.mesh_objs.append(obj)
        
    def get_camera_coods(self):
        Debugging.log("Camera", self.viewer.get_camera_state(self.client))
        
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

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
import viser.transforms as vtf
import viser

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
        self.FOV_width = 50
        self.FOV_height = 50
        self.pixel_area = 1
        self.mesh_objs = []
        self.viewer.viser_server.add_gui_button("Add Density GUI").on_click(lambda _: self.add_gui())
        self.frame_factor = 1
        # self.translate_pos_from_omnivers = (1.1250000243727118, 1.2750000505149364, 3.652500031888485) # diffuse_grey_yellow
        # self.translate_pos_from_omnivers = (0.0, 0.0, 5.0) # cendroid
        # self.translate_pos_from_omnivers = (0.0, 0.0, 1.5714285714285714) # cendroid3
        # self.translate_pos_from_omnivers = (0.0, 0.0, 2.4285714285714284) # cendroid4
        # self.translate_pos_from_omnivers = (0.0, 0.0, 3.0) # cendroid4b
        # self.translate_pos_from_omnivers = (3.000000051657359, 0.9999999701976776, 2.0) # cendroid5
        # self.translate_pos_from_omnivers = (1, 1, 3.5400000005960464) # cendroidcamera
        # self.translate_pos_from_omnivers = (1, 1, 3.5400000005960464) # cendroidcamera
        # self.translate_pos_from_omnivers = (1.0000000049670537, 1.0000000049670537, 1.5) # cendroidcamera2
        # self.translate_pos_from_omnivers = (0.0, 0.0, 3.0) # cendroidcamera3
        # self.translate_pos_from_omnivers = (0, 0, 0) # origin
        # self.translate_pos_from_omnivers = (1, 1, 3.5399999618530273) # yellow_grey
        # self.translate_pos_from_omnivers = (4.238552619995062e-08, 0, 3.5399999618530273) # cube
        # self.translate_pos_from_omnivers = (1.2000001668930054, 0.6000000238418579, -8.100000381469727) # cube3
        # self.translate_pos_from_omnivers = (1, -1.9999998807907104, 8.5) # cube_singe
        self.translate_pos_from_omnivers = (1, -2, 10) # cube_interference
        
        
        # self.densities = []
        # self.density_locations = []
    
    # def void_id(self):
    #     self.densities = []
    #     self.density_locations = []
        
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
            
            # self.densities.extend(outputs["densities"])
            # self.density_locations.extend(outputs["densities_locations"])

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
        frame_factor = self.viewer.viser_server.add_gui_slider("Coordinate Frame factor", -10, 10, 0.01, 1)
        frame_factor.on_update(lambda _: add_pose_coordinate())
        x_omni, y_omni, z_omni = self.translate_pos_from_omnivers
        def add_pose_coordinate():
            self.frame_factor = frame_factor.value
            self.viewer.viser_server.add_frame(
                "pose_frame", 
                True, 
                position=(-x_omni*self.frame_factor, -y_omni*self.frame_factor, -z_omni*self.frame_factor), 
                wxyz=(1.0, 0.0, 0.0, 0.0), 
                axes_length=0.3, 
                axes_radius=0.01
            )
        
        with self.viewer.viser_server.add_gui_folder("Camera Options"):
            self.viewer.viser_server.add_gui_button("Viser Camera To Box", color="violet").on_click(lambda _: self.set_camera_box("viser_box"))
            self.viewer.viser_server.add_gui_button("Box To Viser Camera", color="violet").on_click(lambda _: self.set_camera_box(""))
            # self.viewer.viser_server.add_gui_button("Box To Nerf Camera", color="violet").on_click(lambda _: self.get_camera_coods("box_nerf"))
            # self.viewer.viser_server.add_gui_button("Box To Nerf Camera", color="violet").on_click(lambda _: self.get_camera_coods())
            
        with self.viewer.viser_server.add_gui_folder("Density Options Box"):
            self.viewer.viser_server.add_gui_button("Pointcloud", color="green").on_click(lambda _: self._show_density())
            self.viewer.viser_server.add_gui_button("Pointcloud Clickable (slow)", color="pink").on_click(lambda _: self._show_density(clickable=True))
            self.viewer.viser_server.add_gui_button("Plot Densites", color="indigo").on_click(lambda _: self._show_density(True))
            self.viewer.viser_server.add_gui_button("Clear Point Cloud", color="red").on_click(lambda _: self.delete_point_cloud())
            
        with self.viewer.viser_server.add_gui_folder("Density Settings"):
            self.box_fov = self.viewer.viser_server.add_gui_slider("Box FOV", 0, 360, 1, 60)
            self.box_heigth = self.viewer.viser_server.add_gui_slider("Box Height", 1, 1080, 1, 50)
            self.box_width = self.viewer.viser_server.add_gui_slider("Box Width", 1, 1920, 1, 50)
            self.box_pa = self.viewer.viser_server.add_gui_slider("Pixel Area", 0, 10, 0.1, 1)

        
        self.box = self.viewer.viser_server.add_camera_frustum(name="box", fov=100.0, aspect=1, scale=0.1 ,color=(235, 52, 79), wxyz=(1, 0, 0, 0), position=(-x_omni*self.frame_factor, -y_omni*self.frame_factor, -z_omni*self.frame_factor))

        
        # th_va= 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002
        with self.viewer.viser_server.add_gui_folder("Density Threshold"):
            self.threshold_slider = self.viewer.viser_server.add_gui_slider("Threshold", -1, 1, 0.001, 0)
            
        with self.viewer.viser_server.add_gui_folder("Box Position"):
            self.box_pos_x = self.viewer.viser_server.add_gui_slider("Pos X", -10, 10, 0.01, 0)
            self.box_pos_y = self.viewer.viser_server.add_gui_slider("Pos Y", -10, 10, 0.01, 0)
            self.box_pos_z = self.viewer.viser_server.add_gui_slider("Pos Z (Height)", -20, 20, 0.01, 0)
        
        with self.viewer.viser_server.add_gui_folder("Box WXYZ"):
            self.box_wxyz_x = self.viewer.viser_server.add_gui_slider("Rot X", -180, 180, 0.1, 0)
            self.box_wxyz_y = self.viewer.viser_server.add_gui_slider("Rot Y", -180, 180, 0.1, 0)
            self.box_wxyz_z = self.viewer.viser_server.add_gui_slider("Rot Z", -180, 180, 0.1, 0)
              
        self.box_pos_x.on_update(lambda _: self.update_cube())
        self.box_pos_y.on_update(lambda _: self.update_cube())
        self.box_pos_z.on_update(lambda _: self.update_cube())
        self.box_wxyz_x.on_update(lambda _: self.update_cube())
        self.box_wxyz_y.on_update(lambda _: self.update_cube())
        self.box_wxyz_z.on_update(lambda _: self.update_cube())
        
        self.threshold_slider.on_update(lambda _: setattr(self, "density_threshold", self.threshold_slider.value))
        self.box_fov.on_update(lambda _: setattr(self, "FOV", self.box_fov.value))
        self.box_heigth.on_update(lambda _: setattr(self, "FOV_height", self.box_heigth.value))
        self.box_width.on_update(lambda _: setattr(self, "FOV_width", self.box_width.value))
        self.box_pa.on_update(lambda _: setattr(self, "pixel_area", self.box_pa.value))
    
    def update_cube(self):
        x, y, z = self.translate_pos_from_omnivers
        self.box.wxyz = R.from_euler('xyz', [self.box_wxyz_x.value, self.box_wxyz_y.value, self.box_wxyz_z.value], degrees=True).as_quat()
        self.box.position = (self.box_pos_x.value-x*self.frame_factor, self.box_pos_y.value-y*self.frame_factor, self.box_pos_z.value-z*self.frame_factor)

        
    def _show_density(self, plot_density: bool = False, clickable: bool = False) -> None:
        """Show the density in the viewer

        Args:
            density_location: the density location
        """ 

        Rv = vtf.SO3(wxyz=self.box.wxyz)
        Rv = Rv @ vtf.SO3.from_x_radians(np.pi)
        Rv = torch.tensor(Rv.as_matrix())
        origin = torch.tensor(self.box.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
        c2w = torch.concatenate([Rv, origin[:, None]], dim=1)
        
        import math
        fx_value = self.FOV_width / (2 * math.tan(math.radians(self.FOV / 2)))
        fy_value = self.FOV_height / (2 * math.tan(math.radians(self.FOV / 2)))

        fx = torch.tensor([[fx_value]], device='cuda:0')
        fy = torch.tensor([[fy_value]], device='cuda:0')
        cx = torch.tensor([[self.FOV_width/2]], device='cuda:0')
        cy = torch.tensor([[self.FOV_height/2]], device='cuda:0')

        camera = Cameras(
            camera_to_worlds=c2w,
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
        outputs = self.viewer.get_model().get_outputs_for_camera(camera, pixel_area=self.pixel_area, width=self.FOV_width, height=self.FOV_height)

        # Extrahiere die Dichtewerte und Dichtepositionen
        all_densities = []
        all_density_locations = []
            
        for densities, locations in zip(outputs["densities"], outputs["densities_locations"]):
            all_densities.append(densities)
            all_density_locations.append(locations) 
            

        # filtered_locations = torch.tensor([])
        filtered_locations = []
        Debugging.log("densities_locations_unfiltert", outputs["densities_locations"][0].shape)

        # Berechne globalen Mittelwert und Standardabweichung über alle Dichtewerte
        all_densities = torch.cat([ray_densities for ray_densities in all_densities if ray_densities.numel() > 0])
        all_density_locations = torch.cat([ray_locations for ray_locations in all_density_locations if ray_locations.numel() > 0])

        global_mean = torch.mean(all_densities)
        global_std = torch.std(all_densities)

        for ray_locations, ray_densities in zip(all_density_locations, all_densities):
            if ray_densities.numel() == 0:
                continue

            # standardisieren
            standardized_densities = (ray_densities - global_mean) / global_std
            mask = standardized_densities.squeeze() > self.density_threshold
            Debugging.log("mask", mask)
            # min_value = standardized_densities.min()
            if torch.any(mask):
                first_index = torch.where(mask)[0][0]
                Debugging.log("first_index", ray_locations[first_index].unsqueeze(0))
                filtered_locations.append(ray_locations[first_index].unsqueeze(0))

                # filtered_densities.append(ray_densities[first_index].unsqueeze(0))
            else:
                pass
        
        filtered_locations = torch.cat(filtered_locations)
        
        # #calculate the transmittance points for each point to get the opacity
        # for ray_locations, ray_densities in zip(all_density_locations[0], all_densities[0]):
        #     distances = [self.compute_distance(origin, location) for location in ray_locations]
        #     t_point = self.find_threshold_point(ray_densities, distances)
        #     # Debugging.log("t_point", t_point)
        #     if(t_point != -1):
        #         filtered_locations = torch.cat([filtered_locations, ray_locations[t_point].unsqueeze(0)])
     

        # # filtered_densities = torch.cat(filtered_densities)
        
        filtered_locations = filtered_locations.cpu().numpy()
        # filtered_densities = filtered_densities.cpu().numpy()

        # remaining_indices = self.filter_nearby_indices(filtered_locations)
        # filtered_locations = filtered_locations[remaining_indices]

        
        Debugging.log("densities_locations_filtert", filtered_locations.shape)
        
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
        
        if not plot_density and not clickable:
            if len(self.mesh_objs) > 0:
                self.delete_point_cloud()
                
            obj = self.viewer.viser_server.add_point_cloud(name="density", points=filtered_locations*VISER_NERFSTUDIO_SCALE_RATIO, colors=(255, 0, 255), point_size=0.01, wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0), visible=True)
            self.mesh_objs.append(obj)
            
        if clickable:
            if len(self.mesh_objs) > 0:
                self.delete_point_cloud()
                
            for index, location in enumerate(filtered_locations):
                self.add_point_as_mesh(location, index)
        
    def add_point_as_mesh(self, location, index, scale_factor=1, base_size=0.03, color=(255, 0, 255)):
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
        obj.on_click(lambda _: self.add_distance_modal(location*VISER_NERFSTUDIO_SCALE_RATIO))
        
        self.mesh_objs.append(obj)
        
    def add_distance_modal(self, point):
        """ 
        add a modal to show the distance of a point
        point: point
        """
        
        distance = self.compute_distance(self.box.position, point)
        distance_label = self.viewer.viser_server.add_label("distance_label", f"Distance: {distance:.2f} m", (1, 0, 0, 0), self.box.position)
        # distance_label.label_size = 0.1
        # distance_ray = self.viewer.viser_server.add_gui_modal("Distance")
        point3 = self.box.position + (point - self.box.position) * 0.001
        vertices = np.array([self.box.position, point, point3])
        faces = np.array([[0, 1, 2]])
        distance_ray = self.viewer.viser_server.add_mesh_simple(
            name="line_mesh",
            vertices=vertices,
            faces=faces,
            color=(155, 0, 0),
            wireframe=True,  # Da es nur eine Linie ist, wireframe auf True setzen
            opacity=None,
            material='standard',
            flat_shading=False,
            side='double',
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            visible=True
        )
        distance_ray.on_click(lambda _: distance_ray.remove())

        # with modal:
        #     self.viewer.viser_server.add_gui_markdown(f"Distance: {distance:.2f} m")
        #     # frame_origin = self.viewer.viser_server.add_frame("origin_frame", True, position=self.box.position, axes_length=0.3, axes_radius=0.01)
        #     # frame_target = self.viewer.viser_server.add_frame("frame_target", True, position=point, axes_length=0.3, axes_radius=0.01)
        #     # frame_origin.on_click(lambda _: frame_origin.remove())
        #     # frame_target.on_click(lambda _: frame_target.remove())
        #     self.viewer.viser_server.add_gui_button("Close").on_click(lambda _: modal.close())
        
    def compute_distance(self, a, b):
        """ 
        computes the distance between two points
        a: point a
        b: point b
        returns: distance
        """

        return (np.linalg.norm(np.array(a) - np.array(b)))
    
    def compute_opacity(self, density, distance):
        """ 
        computes the opacity for each point along the ray    
        returns: alpha (opacity)
        """
        alpha = 1 - np.exp(-np.array(density) * np.array(distance))
        return alpha
    
    def find_threshold_point(self, densities, distances):
        """ 
        find the point where the transmittance falls below a threshold
        threshold: threshold value for transmittance (default: 0.01)
        returns: index of the point where the transmittance falls below the threshold. If no point falls below the threshold, the last index is returned.
        """

        alpha = self.compute_opacity(densities, distances)
        T = self.compute_transmission(alpha)
        for i, T_i in enumerate(T):
            if np.all(T_i < self.threshold_slider.value):
                return i
    
        return -1  # no point: return last index
        
    def compute_transmission(self, alpha):
        """ 
        computes the transmission for each point along the ray 
        alpha: opacity
        return: T (transmission)
        """
        T = [1.0]  # init with 1 for the first point
        for i in range(1, len(alpha)):
            T_i = T[-1] * (1 - alpha[i-1])  # T[i] = T[i-1] * (1 - alpha[i-1])
            T.append(T_i)
        return T
    
    def set_camera_box(self, type: str):
        
        clients = self.viewer.viser_server.get_clients()
        for id, client in clients.items():
            if type == "viser_box":
                client.camera.position = self.box.position
                client.camera.wxyz = self.box.wxyz
            else:
                self.box.position = client.camera.position
                self.box.wxyz = client.camera.wxyz
                x, y, z = self.box.position
                q_x, q_y, q_z = R.from_quat(self.box.wxyz).as_euler('xyz', degrees=True)
                self.box_pos_x.value = x
                self.box_pos_y.value = y
                self.box_pos_z.value = z
                self.box_wxyz_x.value = q_x
                self.box_wxyz_y.value = q_y
                self.box_wxyz_z.value = q_z

 
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

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
from scipy.spatial import cKDTree
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
        self.viewer.viser_server.add_gui_button("void it", color="pink").on_click(lambda _: self.void_id())
        self.viewer.viser_server.add_gui_button("Td", color="pink").on_click(lambda _: self._show_density())
        
        #-------------------------------------------------------------
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
                # print("camera: ", camera)
                # print("camera.camera_to_worlds: ", camera.camera_to_worlds)
                R = camera.camera_to_worlds[0, 0:3, 0:3].T
                # print("R: ", R)
                camera_ray_bundle = camera.generate_rays(camera_indices=0, obb_box=obb)
                # print("camera_ray_bundle: ", camera_ray_bundle)
                pts = camera_ray_bundle.directions * outputs["depth"]
                pts = (R @ (pts.view(-1, 3).T)).T.view(*camera_ray_bundle.directions.shape)
                # print("pts: ", pts)
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
            
            self.densities.append(outputs["density"])
            self.density_locations.append(outputs["density_locations"])
            
            print(len(self.densities))
            print(len(self.density_locations))

            self._send_output_to_viewer(outputs, static_render=(action.action in ["static", "step"]))
            # self.viewer.viser_server.add_gui_button("Render in Viser", color="pink").on_click(lambda _: self._show_density())
            # self._show_density(outputs["density"], outputs["density_locations"])

    def filter_nearby_points(self, points, distance_threshold=0.008):
        tree = cKDTree(points)
        kept_indices = np.ones(len(points), dtype=bool)  # Markiert, welche Punkte behalten werden

        for i in range(len(points)):
            if kept_indices[i]:  # Wenn dieser Punkt noch nicht verworfen wurde
                # Finde alle anderen Punkte in der Nähe dieses Punktes, verwende Euklidische Distanz (p=2)
                nearby_indices = tree.query_ball_point(points[i], r=distance_threshold, p=2)
                # Setze alle anderen Punkte in der Nähe auf "nicht behalten", außer den aktuellen Punkt
                kept_indices[nearby_indices] = False
                kept_indices[i] = True  # Stelle sicher, dass dieser Punkt behalten wird

        # Erzeuge den Array der gefilterten Punkte
        filtered_points = points[kept_indices]
        return filtered_points
    
    def _show_density(self):
        """Show the density in the viewer

        Args:
            density_location: the density location
        """
        
        import random
        import string
        import plotly.graph_objects as go
        
        threshold = 0.6
        # print(self.densities[0].shape)
        for i in range(len(self.densities)):
            density = self.densities[i]
            density_location = self.density_locations[i]

            density = density.squeeze().cpu()
            density_location = density_location.cpu()
            mask = density > threshold
            density_location = density_location[mask]
            density_location = density_location.detach().numpy()
            # density = density[mask]
            # density = density.detach().numpy()
            print("in", density_location.shape)
            filtered_density_location = self.filter_nearby_points(density_location)
            print("out",filtered_density_location.shape)
            letters = string.ascii_letters 
            random_string = ''.join(random.choice(letters) for _ in range(3))
            
            for i in range(0, len(filtered_density_location), 10):
                position = (density_location[i][0].item(), density_location[i][1].item(), density_location[i][2].item())
                self.viewer.viser_server.add_icosphere(
                    name=f"{random_string}_point_{i}",
                    subdivisions=1,
                    wxyz=(0, 0, 0, 0),
                    radius=0.008,
                    color=(200, 0, 200),
                    position=position, 
                    visible=True
            )
            
            # normalized_density = (density - density.min()) / (density.max() - density.min())
            
        #     trace = go.Scatter3d(
        #         x=density_location[:, 0],  # X Koordinaten aller Punkte
        #         y=density_location[:, 1],  # Y Koordinaten aller Punkte
        #         z=density_location[:, 2],  # Z Koordinaten aller Punkte
        #         mode='markers',
        #         marker=dict(
        #             size=2,
        #             color=normalized_density,  # Verwendung der normalisierten Dichte als Farbwert
        #             colorscale='Viridis',
        #             opacity=0.8,
        #             colorbar=dict(title='Normalized Density')
        #         )
        #     )
                
        #     # Erstellen des Layouts für den Plot
        #     layout = go.Layout(
        #         title="3D Density Visualization",
        #         scene=dict(
        #             xaxis=dict(title='X'),
        #             yaxis=dict(title='Y'),
        #             zaxis=dict(title='Z')
        #         )
        #     )

        # # Kombinieren von Trace und Layout in einer Figur und Anzeigen des Plots
        # fig = go.Figure(data=[trace], layout=layout)
        # fig.show()

        # for ray_idx, ray in enumerate(density_location):
        #     for sample_idx, sample in enumerate(ray):
        #         sphere_name = f"ray_{ray_idx}_sample_{sample_idx}"
                
        #         print("sample ", sample)
        #         # detach the tensor from the computation graph
        #         sample_detached = sample.detach() # Detach the tensor
        #         # sample_tuple = tuple(sample_detached.numpy()) # Convert to NumPy array then to tuple
        #         # self.viewer.viser_server.add_icosphere(
        #         #     name=sphere_name,
        #         #     subdivisions=2,
        #         #     wxyz= (0, 0, 0, 0),
        #         #     radius=0.005,
        #         #     color=(155, 0, 0),
        #         #     position=sample_tuple,  # Konvertiere NumPy Array zu Tuple
        #         #     visible=True
                # )
                 
    def check_interrupt(self, frame, event, arg):
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.interrupt_render_flag:
                self.interrupt_render_flag = False
                raise viewer_utils.IOChangeException
        return self.check_interrupt

    def _send_output_to_viewer(self, outputs: Dict[str, Any], static_render: bool = True):
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
            dimensions=outputs[output_render].shape[-1], dtype=outputs[output_render].dtype
        )
        selected_output = colormaps.apply_colormap(
            image=outputs[self.viewer.control_panel.output_render],
            colormap_options=self.viewer.control_panel.colormap_options,
        )

        if self.viewer.control_panel.split:
            split_output_render = self.viewer.control_panel.split_output_render
            self.viewer.update_split_colormap_options(
                dimensions=outputs[split_output_render].shape[-1], dtype=outputs[split_output_render].dtype
            )
            split_output = colormaps.apply_colormap(
                image=outputs[self.viewer.control_panel.split_output_render],
                colormap_options=self.viewer.control_panel.split_colormap_options,
            )
            split_index = min(
                int(self.viewer.control_panel.split_percentage * selected_output.shape[1]),
                selected_output.shape[1] - 1,
            )
            selected_output = torch.cat([selected_output[:, :split_index], split_output[:, split_index:]], dim=1)
            selected_output[:, split_index] = torch.tensor([0.133, 0.157, 0.192], device=selected_output.device)

        selected_output = (selected_output * 255).type(torch.uint8)
        depth = (
            outputs["gl_z_buf_depth"].cpu().numpy() * self.viser_scale_ratio if "gl_z_buf_depth" in outputs else None
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
            else 75
            if self.viewer.render_tab_state.preview_render
            else 40
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
            if writer.is_initialized() and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                vis_rays_per_sec = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
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

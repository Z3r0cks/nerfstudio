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

""" Manage the state of the viewer """
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import os
import numpy as np
import torch
import viser
import viser.theme
import viser.transforms as vtf
from typing_extensions import assert_never
#-------------------------------------------------------------
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import plotly.graph_objects as go

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nerfstudio.viewer.control_panel import ControlPanel
from nerfstudio.viewer.export_panel import populate_export_tab
from nerfstudio.viewer.render_panel import populate_render_tab
from nerfstudio.viewer.render_state_machine import RenderAction, RenderStateMachine
from nerfstudio.viewer.utils import CameraState, parse_object
from nerfstudio.viewer.viewer_elements import ViewerControl, ViewerElement
from nerfstudio.viewer_legacy.server import viewer_utils
#-------------------------------------------------------------
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.cameras.rays import Frustums, RaySamples, RayBundle
from mpl_toolkits.mplot3d import Axes3D
from nerfstudio.utils.debugging import Debugging


if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer


VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0


@decorate_all([check_main_thread])
class ViewerDensity:
    """Class to hold state for viewer variables

    Args:
        config: viewer setup configuration
        log_filename: filename to log viewer output to
        datapath: path to data
        pipeline: pipeline object to use
        trainer: trainer object to use
        share: print a shareable URL

    Attributes:
        viewer_info: information string for the viewer
        viser_server: the viser server
    """

    viewer_info: List[str]
    viser_server: viser.ViserServer

    def __init__(
        self,
        config: cfg.ViewerConfig,
        log_filename: Path,
        datapath: Path,
        pipeline: Pipeline,
        # trainer: Optional[Trainer] = None,
        train_lock: Optional[threading.Lock] = None,
        share: bool = False,
    ):
        self.ready = False  # Set to True at end of constructor.
        self.config = config
        # self.trainer = trainer
        self.last_step = 0
        self.train_lock = train_lock
        self.pipeline = pipeline
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath
        self.include_time = self.pipeline.datamanager.includes_time

        if self.config.websocket_port is None:
            websocket_port = viewer_utils.get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        # viewer specific variables
        self.output_type_changed = True
        self.output_split_type_changed = True
        self.step = 0
        self.train_btn_state: Literal["training", "paused", "completed"] = "training"
        self._prev_train_state: Literal["training", "paused", "completed"] = "training"
        self.last_move_time = 0
        
        self.origin_frame_active = False
        self.origin_frame = None

        self.viser_server = viser.ViserServer(host=config.websocket_host, port=websocket_port)
        # Set the name of the URL either to the share link if available, or the localhost
        share_url = None
        if share:
            share_url = self.viser_server.request_share_url()
            if share_url is None:
                print("Couldn't make share URL!")

        if share_url is not None:
            self.viewer_info = [f"Viewer at: http://localhost:{websocket_port} or {share_url}"]
        elif config.websocket_host == "0.0.0.0":
            # 0.0.0.0 is not a real IP address and was confusing people, so
            # we'll just print localhost instead. There are some security
            # (and IPv6 compatibility) implications here though, so we should
            # note that the server is bound to 0.0.0.0!
            self.viewer_info = [f"Viewer running locally at: http://localhost:{websocket_port} (listening on 0.0.0.0)"]
        else:
            self.viewer_info = [f"Viewer running locally at: http://{config.websocket_host}:{websocket_port}"]

        buttons = (
            viser.theme.TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://nerf.studio",
            ),
            viser.theme.TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/nerfstudio-project/nerfstudio",
            ),
            viser.theme.TitlebarButton(
                text="Documentation",
                icon="Description",
                href="https://docs.nerf.studio",
            ),
        )
        image = viser.theme.TitlebarImage(
            image_url_light="https://docs.nerf.studio/_static/imgs/logo.png",
            image_url_dark="https://docs.nerf.studio/_static/imgs/logo-dark.png",
            image_alt="NerfStudio Logo",
            href="https://docs.nerf.studio/",
        )
        titlebar_theme = viser.theme.TitlebarConfig(buttons=buttons, image=image)
        self.viser_server.configure_theme(
            titlebar_content=titlebar_theme,
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )

        self.render_statemachines: Dict[int, RenderStateMachine] = {}
        self.viser_server.on_client_disconnect(self.handle_disconnect)
        self.viser_server.on_client_connect(self.handle_new_client)

        # # Populate the header, which includes the pause button, train cam button, and stats
        # self.pause_train = self.viser_server.add_gui_button(
        #     label="Pause Training", disabled=False, icon=viser.Icon.PLAYER_PAUSE_FILLED
        # )
        
        
        # self.pause_train.on_click(lambda _: self.toggle_pause_button())
        # self.pause_train.on_click(lambda han: self._toggle_training_state(han))
        # self.resume_train = self.viser_server.add_gui_button(
        #     label="Resume Training", disabled=False, icon=viser.Icon.PLAYER_PLAY_FILLED
        # )
        # self.resume_train.on_click(lambda _: self.toggle_pause_button())
        # self.resume_train.on_click(lambda han: self._toggle_training_state(han))
        # self.resume_train.visible = False
        # Add buttons to toggle training image visibility
        # self.hide_images = self.viser_server.add_gui_button(
        #     label="Hide Train Cams", disabled=False, icon=viser.Icon.EYE_OFF, color=None
        # )
        # self.hide_images.on_click(lambda _: self.set_camera_visibility(False))
        # self.hide_images.on_click(lambda _: self.toggle_cameravis_button())
        # self.show_images = self.viser_server.add_gui_button(
        #     label="Show Train Cams", disabled=False, icon=viser.Icon.EYE, color=None
        # )
        # self.show_images.on_click(lambda _: self.set_camera_visibility(True))
        # self.show_images.on_click(lambda _: self.toggle_cameravis_button())
        # self.show_images.visible = False
        mkdown = self.make_stats_markdown(0, "0x0px")
        self.stats_markdown = self.viser_server.add_gui_markdown(mkdown)
        
        #------------------------------------------------------
        self.get_density_button = self.viser_server.add_gui_button(
            label="Visualize 3D Scene", disabled=False, color="cyan", 
        )
        self.get_density_button.on_click(lambda _: self.get_density("3d"))
        
        self.visualize_density_button = self.viser_server.add_gui_button(
            label="Histogram", disabled=False, color="yellow", 
        )
        self.visualize_density_button.on_click(lambda _: self.get_density("his"))
        
        self.visualize_density_viser_button = self.viser_server.add_gui_button(
            label="Render in Viser", disabled=False, color="pink", 
        )
        self.visualize_density_viser_button.on_click(lambda _: self.get_density("viser"))
        
        self.viser_button = self.viser_server.add_gui_button(
            label="Toggle Origin Frame", disabled=False, color="red", 
        )
        self.viser_button.on_click(lambda _: self.viser_example())
        
        #------------------------------------------------------
        
        tabs = self.viser_server.add_gui_tab_group()
        control_tab = tabs.add_tab("Control", viser.Icon.SETTINGS)
        with control_tab:
            self.control_panel = ControlPanel(
                self.viser_server,
                self.include_time,
                VISER_NERFSTUDIO_SCALE_RATIO,
                self._trigger_rerender,
                self._output_type_change,
                self._output_split_type_change,
                default_composite_depth=self.config.default_composite_depth,
            )
        config_path = self.log_filename.parents[0] / "config.yml"
        with tabs.add_tab("Render", viser.Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                self.viser_server, config_path, self.datapath, self.control_panel
            )

        # with tabs.add_tab("Export", viser.Icon.PACKAGE_EXPORT):
        #     populate_export_tab(self.viser_server, self.control_panel, config_path, self.pipeline.model)

        # Keep track of the pointers to generated GUI folders, because each generated folder holds a unique ID.
        viewer_gui_folders = dict()
        
        def nested_folder_install(folder_labels: List[str], prev_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [prev_cb(element), self._trigger_rerender()]
            else:
                # recursively create folders
                # If the folder name is "Custom Elements/a/b", then:
                #   in the beginning: folder_path will be
                #       "/".join([] + ["Custom Elements"]) --> "Custom Elements"
                #   later, folder_path will be
                #       "/".join(["Custom Elements"] + ["a"]) --> "Custom Elements/a"
                #       "/".join(["Custom Elements", "a"] + ["b"]) --> "Custom Elements/a/b"
                #  --> the element will be installed in the folder "Custom Elements/a/b"
                #
                # Note that the gui_folder is created only when the folder is not in viewer_gui_folders,
                # and we use the folder_path as the key to check if the folder is already created.
                # Otherwise, use the existing folder as context manager.
                folder_path = "/".join(prev_labels + [folder_labels[0]])
                if folder_path not in viewer_gui_folders:
                    viewer_gui_folders[folder_path] = self.viser_server.add_gui_folder(folder_labels[0])
                with viewer_gui_folders[folder_path]:
                    nested_folder_install(folder_labels[1:], prev_labels + [folder_labels[0]], element)

        with control_tab:
            from nerfstudio.viewer_legacy.server.viewer_elements import ViewerElement as LegacyViewerElement

            if len(parse_object(pipeline, LegacyViewerElement, "Custom Elements")) > 0:
                from nerfstudio.utils.rich_utils import CONSOLE

                CONSOLE.print(
                    "Legacy ViewerElements detected in model, please import nerfstudio.viewer.viewer_elements instead",
                    style="bold yellow",
                )
            self.viewer_elements = []
            self.viewer_elements.extend(parse_object(pipeline, ViewerElement, "Custom Elements"))
            for param_path, element in self.viewer_elements:
                folder_labels = param_path.split("/")[:-1]
                nested_folder_install(folder_labels, [], element)

            # scrape the trainer/pipeline for any ViewerControl objects to initialize them
            self.viewer_controls: List[ViewerControl] = [
                e for (_, e) in parse_object(pipeline, ViewerControl, "Custom Elements")
            ]
        for c in self.viewer_controls:
            c._setup(self)

        # Diagnostics for Gaussian Splatting: where the points are at the start of training.
        # This is hidden by default, it can be shown from the Viser UI's scene tree table.
        if isinstance(pipeline.model, SplatfactoModel):
            self.viser_server.add_point_cloud(
                "/gaussian_splatting_initial_points",
                points=pipeline.model.means.numpy(force=True) * VISER_NERFSTUDIO_SCALE_RATIO,
                colors=(255, 0, 0),
                point_size=0.01,
                point_shape="circle",
                visible=False,  # Hidden by default.
            )
        self.ready = True

    #------------------------------------------------------ 
    def viser_example(self) -> None:
        
        self.viser_server.world_axes.visible = True
        self.viser_server.add_icosphere("/first", 0.2, (255, 0, 0), 3, (0,0,0,0), (0,0,0))
        self.viser_server.add_icosphere("/second", 0.2, (255, 0, 0), 3, (0,0,0,0), (1,0,0))
        self.viser_server.add_icosphere("/third", 0.2, (255, 0, 0), 3, (0,0,0,0), (2,0,0))
        self.viser_server.add_icosphere("/fourth", 0.2, (255, 0, 0), 3, (0,0,0,0), (3,0,0))
        self.viser_server.add_icosphere("/fifth", 0.2, (255, 0, 0), 3, (0,0,0,0), (4,0,0))

        # server = viser.ViserServer()

        # while True:
        #     # Add some coordinate frames to the scene. These will be visualized in the viewer.
        # Debugging.write_to_file(self.viser_server, "viser_server")
        
        # testframe = self.viser_server.add_frame(
        #     "/tree/branch",
        #     wxyz=(1.0, 0.0, 0.0, 0.0),
        #     position=(0.0, 0.0, 0.0),
        # )
        # self.test_viser_frame = True
        
        # if self.origin_frame_active == False:
        #     self.origin_frame = self.viser_server.add_frame(
        #         "/tree/branch",
        #         wxyz=(0.0, 0.0, 0.0, 0.0),
        #         position=(self.viser_server.get_clients()[0].camera._state.position),
        #     )
        #     self.origin_frame_active = True
        # else:
        #     self.origin_frame.visible = False
        #     self.origin_frame_active = False
        # self.viser_server.world_axes.position = (3.0, 12.0, -5.0)
        # self.viser_server.world_axes.visible = True
        # self.viser_server.set_up_direction((0.0, 0.0, 0.0))
        
        # self.viser_server.get_clients()
        # Debugging.write_to_file(self.viser_server.get_clients(), "viser_server.get_clients")
        
        # self.viser_server.get
        # print(viser.SceneNodePointerEvent[source]
        
        
    
    def get_density(self, button_type: str) -> None:
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_rays = 20
        # examble
        origins = torch.zeros((num_rays, 3), device=device)  # 10 rays, 3 dimensions (X, Y, Z)
        directions = torch.rand((num_rays, 3), device=device)  # 10 direction vectors (normalized)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)  # direction vectors normalizing
        pixel_area = torch.ones((num_rays, 1), device=device)  # area of pixel with 1 m distance from origin
        
        # optional attriutes
        camera_indices = torch.randint(low=0, high=5, size=(num_rays, 1), device=device)
        nears = torch.zeros((num_rays, 1), device=device)  # begin from sampling alont the rays
        fars = torch.ones((num_rays, 1), device=device) * 2  # end
        
        # num_rays = origins.size(0)
        num_samples = int(fars.max().item() / 0.1) # samplerate  nt(fars.max().item() / 0.1) = 1500 (0.1 meter)
        
        # generate times that for every ray identical times are used
        
        # # examble
        # origins = torch.zeros((10, 3), device=device)  # 10 rays, 3 dimensions (X, Y, Z)
        # directions = torch.rand((10, 3), device=device)  # 10 direction vectors (normalized)
        # directions = directions / torch.norm(directions, dim=1, keepdim=True)  # direction vectors normalizing
        # pixel_area = torch.ones((10, 1), device=device)  # area of pixel with 1 m distance from origin
        
        # # optional attriutes
        # camera_indices = torch.randint(low=0, high=5, size=(10, 1), device=device)
        # nears = torch.zeros((10, 1), device=device)  # begin from sampling alont the rays
        # fars = torch.ones((10, 1), device=device) * 150  # end
        times = torch.linspace(0, num_samples, 10, device=device).view(-1, 1)  # time for sampling

        # create ray bundle
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            nears=nears,
            fars=fars,
            # times=times
        )

        # # create ray bundle
        # ray_bundle = RayBundle(
        #     origins=origins,
        #     directions=directions,
        #     pixel_area=pixel_area,
        #     camera_indices=camera_indices,
        #     nears=nears.repeat(1, num_samples).view(10, num_samples, 1),
        #     fars=fars.repeat(1, num_samples).view(10, num_samples, 1),
        #     times=torch.linspace(0, 1, 10, device=device).view(-1, 1)
        # )
        
        model = self.pipeline.model.to(device)
        outputs = model.get_outputs(ray_bundle)
        density = torch.Tensor(outputs["density"])  # Convert output to torch.Tensor
        # print(density.shape): torch.Size([10, 48, 1])
        Debugging.write_to_file(density, "density_six_rays")
        if button_type == "his":
            self.visualize_density_histogram(density)
        elif button_type == "3d":
            self.visualize_density_3d(origins, directions, density)
        else:
            self.visualize_density_viser(ray_bundle, density)
            
    def visualize_density_viser(self, ray_bundle: RayBundle, density: torch.Tensor) -> None:
        
        print("Visualizing density in Viser...")
        device = density.device
        density = density.squeeze()  # remove singular dimension
        max_density = density.max()
        density_normalized = density / max_density
        # end_value = ray_bundle.fars[0].item()
        
        # ray_length = torch.linspace(0, end_value, density.shape[1], device=device)  # generate line space along the ray
        
        # find the first point where density exceeds the threshold
        density_threshold = 0.3
        threshold_mask = density_normalized > density_threshold
        first_exceeds = threshold_mask.int().argmax(dim=1)  # get the index of the first exceedance
        
        # ensure you consider cases where no points exceed by checking the condition
        valid_mask = threshold_mask.any(dim=1)
        first_exceeds[~valid_mask] = density.shape[1] - 1  # use the last point if no exceedance
        
        # filter all points after the threshold exceedance
        valid_points_mask = torch.arange(density.shape[1], device=device)[None, :] <= first_exceeds[:, None]
        
        # calculate the ray points only up to the first exceedance
        for i in range(ray_bundle.origins.shape[0]):  # iterate over all rays
            if ray_bundle.nears is not None and ray_bundle.fars is not None:
                near = ray_bundle.nears[i].item()
                far = ray_bundle.fars[i].item()
                ray_length = torch.linspace(near, far, density.shape[1], device=device)
            
            for j in range(density.shape[1]):  # iterate over points in each ray
                if valid_points_mask[i, j]:
                    point = ray_bundle.origins[i] + ray_bundle.directions[i] * ray_length[j]
                    normalized_density = density_normalized[i, j].item()
                    
                    # Define the color based on density, converting it to RGB
                    color_intensity = int(255 * normalized_density)
                    color = (color_intensity, 0, 0)  # Red color intensity based on density
                    
                    # Add sphere at this point
                    sphere_name = f"/ray_{i}_point_{j}"
                    self.viser_server.add_icosphere(
                        name=sphere_name,
                        radius=0.1  ,  # smaller radius for visualization
                        color=color,
                        subdivisions=2,
                        wxyz=(1.0, 0.0, 0.0, 0.0),  # default orientation
                        position=point.cpu().detach().numpy(),
                        visible=True
                    )
    
    def visualize_density_histogram(self, density: torch.Tensor) -> None:
        # Density histogram along the rays
        density = density.cpu().detach().numpy().squeeze()
        fig = go.Figure()
        for i, single_ray_density in enumerate(density):
            fig.add_trace(go.Scatter(y=single_ray_density, mode='lines', name=f'Ray {i}'))        
        fig.update_layout(title='Density along rays', 
                          xaxis_title='Sample along ray', 
                          yaxis_title='Density',
                          legend_title='Ray Index')
        fig.show()
        
    def visualize_density_3d(self, origins, direction, density: torch.Tensor) -> None:
        density_threshold = 0.3
        device = density.device
        density = density.squeeze() #remove singlar dimension
        max_density = density.max()
        density_normalized = density / max_density
        
        ray_length = torch.linspace(0, 1, density.shape[1], device=device) # generate line space along the ray
        ray_points = origins[:, None, :] + direction[:, None, :] * ray_length[:, None] # generate points along the ray

        # find the first point where density exceeds the threshold
        threshold_mask = density_normalized > density_threshold
        first_exceeds = threshold_mask.int().argmax(dim=1)  # get the index of the first exceedance
        
        # ensure you consider cases where no points exceed by checking the condition
        valid_mask = threshold_mask.any(dim=1)
        first_exceeds[~valid_mask] = density.shape[1] - 1  # use the last point if no exceedance
        
        # filter all points after the threshold exceedance
        valid_points_mask = torch.arange(density.shape[1], device=device)[None, :] <= first_exceeds[:, None]
        
        # calculate the ray points only up to the first exceedance
        ray_points = origins[:, None, :] + direction[:, None, :] * ray_length[:, None]
        ray_points = ray_points[valid_points_mask]
        density_normalized = density_normalized[valid_points_mask]
        
        # ürepare data for plotting
        x, y, z = ray_points[..., 0].cpu().numpy(), ray_points[..., 1].cpu().numpy(), ray_points[..., 2].cpu().numpy()
        color = density_normalized.detach().cpu().numpy() # color based on density
        
        # create 3d plot
        trace = go.Scatter3d(
            x=x.ravel(), y=y.ravel(), z=z.ravel(), # flatten the points
            mode='markers',
            marker = dict(
                size=2,
                color=color.ravel(),
                colorscale='Viridis',
                opacity =.8,
                colorbar=dict(title='Normalized Density')
            )
        )
        
        layout = go.Layout(
            title="3D Density Visualization",
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            )
        )
        
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()
    
    #------------------------------------------------------
    
    # def toggle_pause_button(self) -> None:
    #     self.pause_train.visible = not self.pause_train.visible
    #     self.resume_train.visible = not self.resume_train.visible

    # def toggle_cameravis_button(self) -> None:
    #     self.hide_images.visible = not self.hide_images.visible
    #     self.show_images.visible = not self.show_images.visible

    def make_stats_markdown(self, step: Optional[int], res: Optional[str]) -> str:
        # if either are None, read it from the current stats_markdown content
        if step is None:
            step = int(self.stats_markdown.content.split("\n")[0].split(": ")[1])
        if res is None:
            res = (self.stats_markdown.content.split("\n")[1].split(": ")[1]).strip()
        return f"Step: {step}  \nResolution: {res}"

    def update_step(self, step):
        """
        Args:
            step: the train step to set the model to
        """
        self.stats_markdown.content = self.make_stats_markdown(step, None)

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        R = vtf.SO3(wxyz=client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(client.camera.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        if self.ready and self.render_tab_state.preview_render:
            camera_type = self.render_tab_state.preview_camera_type
            camera_state = CameraState(
                fov=self.render_tab_state.preview_fov,
                aspect=self.render_tab_state.preview_aspect,
                c2w=c2w,
                time=self.render_tab_state.preview_time,
                camera_type=CameraType.PERSPECTIVE
                if camera_type == "Perspective"
                else CameraType.FISHEYE
                if camera_type == "Fisheye"
                else CameraType.EQUIRECTANGULAR
                if camera_type == "Equirectangular"
                else assert_never(camera_type),
            )
        else:
            camera_state = CameraState(
                fov=client.camera.fov,
                aspect=client.camera.aspect,
                c2w=c2w,
                camera_type=CameraType.PERSPECTIVE,
            )
        return camera_state

    def handle_disconnect(self, client: viser.ClientHandle) -> None:
        self.render_statemachines[client.client_id].running = False
        self.render_statemachines.pop(client.client_id)

    def handle_new_client(self, client: viser.ClientHandle) -> None:
        self.render_statemachines[client.client_id] = RenderStateMachine(self, VISER_NERFSTUDIO_SCALE_RATIO, client)
        self.render_statemachines[client.client_id].start()
        

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            if not self.ready:
                return
            self.last_move_time = time.time()
            with self.viser_server.atomic():
                camera_state = self.get_camera_state(client)
                self.render_statemachines[client.client_id].action(RenderAction("move", camera_state))

    # def set_camera_visibility(self, visible: bool) -> None:
    #     """Toggle the visibility of the training cameras."""
    #     with self.viser_server.atomic():
    #         for idx in self.camera_handles:
    #             self.camera_handles[idx].visible = visible

    # def update_camera_poses(self):
    #     # TODO this fn accounts for like ~5% of total train time
    #     # Update the train camera locations based on optimization
    #     assert self.camera_handles is not None
    #     if hasattr(self.pipeline.datamanager, "train_camera_optimizer"):
    #         camera_optimizer = self.pipeline.datamanager.train_camera_optimizer
    #     elif hasattr(self.pipeline.model, "camera_optimizer"):
    #         camera_optimizer = self.pipeline.model.camera_optimizer
    #     else:
    #         return
    #     idxs = list(self.camera_handles.keys())
    #     with torch.no_grad():
    #         assert isinstance(camera_optimizer, CameraOptimizer)
    #         c2ws_delta = camera_optimizer(torch.tensor(idxs, device=camera_optimizer.device)).cpu().numpy()
    #     for i, key in enumerate(idxs):
    #         # both are numpy arrays
    #         c2w_orig = self.original_c2w[key]
    #         c2w_delta = c2ws_delta[i, ...]
    #         c2w = c2w_orig @ np.concatenate((c2w_delta, np.array([[0, 0, 0, 1]])), axis=0)
    #         R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
    #         R = R @ vtf.SO3.from_x_radians(np.pi)
    #         self.camera_handles[key].position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
    #         self.camera_handles[key].wxyz = R.wxyz

    def _trigger_rerender(self) -> None:
        """Interrupt current render."""
        if not self.ready:
            return
        clients = self.viser_server.get_clients()
        for id in clients:
            camera_state = self.get_camera_state(clients[id])
            self.render_statemachines[id].action(RenderAction("move", camera_state))

    # def _toggle_training_state(self, _) -> None:
    #     """Toggle the trainer's training state."""
    #     if self.trainer is not None:
    #         if self.trainer.training_state == "training":
    #             self.trainer.training_state = "paused"
    #         elif self.trainer.training_state == "paused":
    #             self.trainer.training_state = "training"

    def _output_type_change(self, _):
        self.output_type_changed = True

    def _output_split_type_change(self, _):
        self.output_split_type_changed = True

    def _pick_drawn_image_idxs(self, total_num: int) -> list[int]:
        """Determine indicies of images to display in viewer.

        Args:
            total_num: total number of training images.

        Returns:
            List of indices from [0, total_num-1].
        """
        if self.config.max_num_display_images < 0:
            num_display_images = total_num
        else:
            num_display_images = min(self.config.max_num_display_images, total_num)
        # draw indices, roughly evenly spaced
        return np.linspace(0, total_num - 1, num_display_images, dtype=np.int32).tolist()

    # def init_scene(
    #     self,
    #     train_dataset: InputDataset,
    #     train_state: Literal["training", "paused", "completed"],
    #     eval_dataset: Optional[InputDataset] = None,
    # ) -> None:
    #     """Draw some images and the scene aabb in the viewer.

    #     Args:
    #         dataset: dataset to render in the scene
    #         train_state: Current status of training
    #     """
        #viewer_state.init_scene(
            # train_dataset=pipeline.datamanager.train_dataset,
            # train_state="completed",
            # eval_dataset=pipeline.datamanager.eval_dataset,
        #)
        # print(train_dataset[0]["image"])
        # open("C:/Users/tkasepa/Desktop/Thesisinhalte/pipeline/viewer/image_indices.txt", "w").write(str(image_indices))
        
        # draw the training cameras and images
        # self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        # self.original_c2w: Dict[int, np.ndarray] = {}
        # image_indices = self._pick_drawn_image_idxs(len(train_dataset))
        # for idx in image_indices: #for every image in the dataset
        #     image = train_dataset[idx]["image"] # image matrix with normalized rgba pixel values
        #     camera = train_dataset.cameras[idx]
        #     image_uint8 = (image * 255).detach().type(torch.uint8)
        #     image_uint8 = image_uint8.permute(2, 0, 1)

        #     # torchvision can be slow to import, so we do it lazily.
        #     import torchvision

        #     image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)  # type: ignore
        #     image_uint8 = image_uint8.permute(1, 2, 0)
        #     image_uint8 = image_uint8.cpu().numpy()
        #     c2w = camera.camera_to_worlds.cpu().numpy()
        #     R = vtf.SO3.from_matrix(c2w[:3, :3])
        #     R = R @ vtf.SO3.from_x_radians(np.pi)
        #     camera_handle = self.viser_server.add_camera_frustum(
        #         name=f"/cameras/camera_{idx:05d}",
        #         fov=float(2 * np.arctan(camera.cx / camera.fx[0])),
        #         scale=self.config.camera_frustum_scale,
        #         aspect=float(camera.cx[0] / camera.cy[0]),
        #         image=image_uint8,
        #         wxyz=R.wxyz,
        #         position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
        #     )

        #     @camera_handle.on_click
        #     def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
        #         with event.client.atomic():
        #             event.client.camera.position = event.target.position
        #             event.client.camera.wxyz = event.target.wxyz

        #     self.camera_handles[idx] = camera_handle
        #     self.original_c2w[idx] = c2w

        # self.train_state = train_state
        # self.train_util = 0.9

    # def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None) -> None:
    #     """updates the scene based on the graph weights

    #     Args:
    #         step: iteration step of training
    #         num_rays_per_batch: number of rays per batch, used during training
    #     """
    #     self.step = step

    #     if len(self.render_statemachines) == 0:
    #         return
    #     # this stops training while moving to make the response smoother
    #     while time.time() - self.last_move_time < 0.1:
    #         time.sleep(0.05)
    #     if self.trainer is not None and self.trainer.training_state == "training" and self.train_util != 1:
    #         if (
    #             EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
    #             and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
    #         ):
    #             train_s = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
    #             vis_s = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
    #             train_util = self.train_util
    #             vis_n = self.control_panel.max_res**2
    #             train_n = num_rays_per_batch
    #             train_time = train_n / train_s
    #             vis_time = vis_n / vis_s

    #             render_freq = train_util * vis_time / (train_time - train_util * train_time)
    #         else:
    #             render_freq = 30
    #         if step > self.last_step + render_freq:
    #             self.last_step = step
    #             clients = self.viser_server.get_clients()
    #             for id in clients:
    #                 camera_state = self.get_camera_state(clients[id])
    #                 if camera_state is not None:
    #                     self.render_statemachines[id].action(RenderAction("step", camera_state))
    #             self.update_camera_poses()
    #             self.update_step(step)

    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_type_changed:
            self.control_panel.update_colormap_options(dimensions, dtype)
            self.output_type_changed = False

    def update_split_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_split_type_changed:
            self.control_panel.update_split_colormap_options(dimensions, dtype)
            self.output_split_type_changed = False

    def get_model(self) -> Model:
        """Returns the model."""
        return self.pipeline.model

    def training_complete(self) -> None:
        """Called when training is complete."""
        self.training_state = "completed"

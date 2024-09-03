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
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.viewer.control_panel import ControlPanel
from nerfstudio.viewer.render_panel import populate_render_tab
from nerfstudio.viewer.render_state_machine_lidar import RenderAction, RenderStateMachine
from nerfstudio.viewer.utils import CameraState, parse_object
from nerfstudio.viewer.viewer_elements import ViewerControl, ViewerElement
from nerfstudio.viewer_legacy.server import viewer_utils
from nerfstudio.viewer.render_state_machine_lidar import RenderAction

# if TYPE_CHECKING:
#     from nerfstudio.engine.trainer import Trainer

VISER_NERFSTUDIO_SCALE_RATIO: float = 1.0


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
        dataparser_transforms_path: str,
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
        self.dataparser_transforms_path = dataparser_transforms_path
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

        mkdown = self.make_stats_markdown(0, "0x0px")
        self.stats_markdown = self.viser_server.add_gui_markdown(mkdown)
        
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

        viewer_gui_folders = dict()
        
        def nested_folder_install(folder_labels: List[str], prev_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [prev_cb(element), self._trigger_rerender()]
            else:
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


    def _trigger_rerender(self) -> None:
        """Interrupt current render."""
        if not self.ready:
            return
        clients = self.viser_server.get_clients()
        for id in clients:
            camera_state = self.get_camera_state(clients[id])
            self.render_statemachines[id].action(RenderAction("move", camera_state))

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

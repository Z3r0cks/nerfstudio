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

#!/usr/bin/env python
"""
Starts viewer in eval mode.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

import tyro

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
# from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer_density import ViewerDensity as ViewerState
from nerfstudio.utils.debugging import Debugging

@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewer:
    """Load a checkpoint and start the viewer."""

    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""
    vis: Literal["viewer", "viewer_legacy"] = "viewer"
    """Type of viewer"""

    def main(self) -> None:
        """Main function."""
        config, pipeline, _, step = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        
        path = str(self.load_config)
        dataparser_transforms_path = f'{str(path[slice(-10)])}dataparser_transforms.json'

        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = self.vis #viewer
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk  
            # ViewerConfig:
            # camera_frustum_scale: 0.1
            # default_composite_depth: True
            # image_format: jpeg
            # jpeg_quality: 75
            # make_share_url: False
            # max_num_display_images: 512
            # num_rays_per_chunk: 32768
            # quit_on_train_completion: False
            # relative_log_filename: viewer_log_filename.txt
            # websocket_host: 0.0.0.0
            # websocket_port: None
            # websocket_port_default: 7007      

        _start_viewer(config, pipeline, step, dataparser_transforms_path)

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """


def _start_viewer(config: TrainerConfig, pipeline: Pipeline, step: int, dataparser_transforms_path: str) -> None:
    """Starts the viewer

    Args:
        config: Configuration of pipeline to load
        pipeline: Pipeline instance of which to load weights
        step: Step at which the pipeline was saved
    """
    
    base_dir = config.get_base_dir()
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    viewer_state = None

    if config.vis == "viewer":
        viewer_state = ViewerState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            share=config.viewer.make_share_url,
            dataparser_transforms_path = dataparser_transforms_path
        )
    else:
        raise ValueError(f"Unknown vis type: {config.vis}")

    while True:
        time.sleep(0.01)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    
    # tyro is a python biblotek that provides a command line interface for python scripts
    tyro.extras.set_accent_color("green")
    tyro.cli(tyro.conf.FlagConversionOff[RunViewer]).main() #type: ignore

if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(tyro.conf.FlagConversionOff[RunViewer])  # noqa
